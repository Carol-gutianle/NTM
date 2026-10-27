"""Stage one HF model copy before torchrun. Standard library only; no GPU work."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LOCAL_FS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs"}
GIB = 1024**3


def log(message):
    print(f"[model cache] {message}", file=sys.stderr, flush=True)


def filesystem(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    try:
        result = subprocess.run(
            ["findmnt", "-T", str(path), "-n", "-o", "FSTYPE"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def inventory(source):
    source = Path(source).resolve()
    if not (source / "config.json").is_file():
        raise ValueError(f"Missing config.json: {source}")
    index = source / "model.safetensors.index.json"
    weights = (set(json.loads(index.read_text())["weight_map"].values())
               if index.exists() else {"model.safetensors"})
    if not weights:
        raise ValueError("Empty weight manifest")
    for name in weights:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe shard name: {name}")
        if not (source / name).is_file():
            raise ValueError(f"Missing shard: {source / name}")
    # Do not recursively copy unrelated checkpoints, caches, credentials or code.
    names = weights | {
        path.name for path in source.iterdir()
        if path.is_file() and not path.name.startswith(".")
        and path.suffix in {".json", ".txt", ".model", ".tiktoken", ".jinja"}
    }
    files = {}
    for name in sorted(names):
        stat = (source / name).stat()
        files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"source": str(source), "files": files}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as file:
        json.dump(data, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def stage(source, root, reserve_bytes=8 * GIB, verify=False, allow_unverified=False):
    spec = inventory(source)
    source = Path(spec["source"])
    root = Path(root).expanduser().resolve()
    if root == source or source in root.parents:
        raise ValueError("Cache root must not be inside the source model directory.")
    fs = filesystem(root)
    if fs not in LOCAL_FS:
        if not allow_unverified:
            raise ValueError(f"{root}: filesystem={fs}, not verified local disk. "
                             "Set LOCAL_MODEL_CACHE to an ext4/xfs local-disk directory. "
                             "For a platform-confirmed local overlay only, explicitly set "
                             "LOCAL_CACHE_ALLOW_UNVERIFIED=1.")
        log(f"WARNING: explicitly accepting filesystem={fs}; locality/speed NOT verified.")
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]
    target = root / f"{source.name}-{key}"
    total = sum(info["size"] for info in spec["files"].values())
    log(f"Source={source}; cache={target}; fs={fs}; weights/assets={total/GIB:.2f} GiB")
    with (root / f".{key}.lock").open("a") as lock:
        log("Waiting for cache lock (only one process copies this model).")
        fcntl.flock(lock, fcntl.LOCK_EX)
        target.mkdir(exist_ok=True)
        manifest_path = target / ".stage-manifest.json"
        try:
            previous = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            previous = {}
        records = previous.get("copied", {}) if previous.get("inventory") == spec else {}
        pending = []
        for name, info in spec["files"].items():
            dest = target / name
            record = records.get(name, {})
            valid = (dest.is_file() and dest.stat().st_size == info["size"]
                     and dest.stat().st_mtime_ns == record.get("mtime_ns")
                     and bool(record.get("sha256")))
            if valid and verify:
                valid = digest(dest) == record["sha256"]
            if not valid:
                pending.append(name)
        remaining = sum(spec["files"][name]["size"] for name in pending)
        free = shutil.disk_usage(root).free
        if pending and free < remaining + reserve_bytes:
            raise ValueError(f"Insufficient cache space: free={free/GIB:.2f} GiB, "
                             f"copy={remaining/GIB:.2f} GiB + reserve={reserve_bytes/GIB:.2f} GiB. "
                             "No automatic deletion of other caches.")
        if pending:
            (target / ".stage-complete.json").unlink(missing_ok=True)
        for number, name in enumerate(pending, 1):
            start = time.monotonic()
            src, dest = source / name, target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name + ".stage-part")
            size, last = 0, start
            h = hashlib.sha256()
            log(f"Copy {number}/{len(pending)}: {name}")
            with src.open("rb") as reader, partial.open("wb") as writer:
                for block in iter(lambda: reader.read(8 * 1024**2), b""):
                    writer.write(block)
                    h.update(block)
                    size += len(block)
                    now = time.monotonic()
                    if now - last >= 10:
                        log(f"{name}: {size/GIB:.2f}/{spec['files'][name]['size']/GIB:.2f} GiB, "
                            f"{size/1024**2/max(now-start, .001):.1f} MiB/s")
                        last = now
                writer.flush()
                os.fsync(writer.fileno())
            stat = src.stat()
            if (size != spec["files"][name]["size"]
                    or stat.st_size != size
                    or stat.st_mtime_ns != spec["files"][name]["mtime_ns"]):
                raise ValueError(f"Source changed while copying: {src}. Retry after publication finishes.")
            partial.replace(dest)
            records[name] = {"sha256": h.hexdigest(), "mtime_ns": dest.stat().st_mtime_ns}
            atomic_json(manifest_path, {"inventory": spec, "copied": records})
            log(f"Copied {name} in {time.monotonic()-start:.1f}s")
        if inventory(source) != spec:
            raise ValueError("Source inventory changed during staging; refusing to launch.")
        atomic_json(target / ".stage-complete.json", {"key": key, "source": str(source)})
        log(f"{'Staged' if pending else 'Reusing complete cache'}: {target}")
        return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache-root", default="auto")
    p.add_argument("--reserve-gib", type=float, default=8)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--allow-unverified-fs", action="store_true")
    args = p.parse_args()
    if args.reserve_gib < 0:
        p.error("reserve-gib must be nonnegative")
    try:
        spec = inventory(args.source)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f"[model cache] ERROR: {exc}\n")
    if args.cache_root == "auto":
        # Only use existing candidate mounts; never assume /tmp is NVMe.
        bases = [Path(x) for x in (os.environ.get("TMPDIR", "/tmp"),
                                  "/local_nvme", "/local", "/scratch", "/tmp")]
        required = sum(x["size"] for x in spec["files"].values())
        bases = [x for x in bases if x.is_dir() and filesystem(x) in LOCAL_FS
                 and os.access(x, os.W_OK)]
        # Prefer an existing per-user cache (it may need no further copy).
        roots = [x / f"lora-model-cache-{os.getuid()}" for x in bases]
        root = next((x for x in roots if x.exists()), None)
        if root is None:
            root = next((x for x in roots
                         if shutil.disk_usage(x.parent).free >= required + args.reserve_gib * GIB), None)
        if root is None:
            p.error("No verified local disk with enough space. Set LOCAL_MODEL_CACHE explicitly "
                    "to a node-local disk, or LOCAL_MODEL_CACHE=off to knowingly use shared storage.")
    else:
        root = Path(args.cache_root)
    try:
        result = stage(args.source, root, int(args.reserve_gib * GIB), args.verify,
                       args.allow_unverified_fs)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f"[model cache] ERROR: {exc}\n")
    print(result, flush=True)  # stdout is reserved for the resulting path.


if __name__ == "__main__":
    main()
