"""Fetch, inspect and run the pinned Kaggle LLM Decoder Qwen v8 notebook.

No model downloads, package installation, training or submission are automatic.
The original notebook is executed only by the explicit `run` subcommand.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "artifacts/decoder_qwen_v8"
SOURCE_URL = "https://www.kaggle.com/api/v1/kernels/pull/lynnsakurai/llm-decoder-qwen-ver"
SOURCE_SHA256 = "06a0aa8e8dfbadb5844db463da3d09c201f76603808414e079e264fa79338079"
BASES = {
    "Qwen3-8B": ("qwen3", 4096),
    "Qwen3-14B": ("qwen3", 5120),
    "Qwen3-4B": ("qwen3", 2560),
    "Qwen2.5-7B": ("qwen2", 3584),
    "GLM4-9B": ("glm4", 4096),
    "Qwen3.5 token-attention base": ("qwen3_5", None),
}
PACKAGES = [
    "torch", "transformers", "peft", "accelerate", "numpy", "pandas",
    "scipy", "scikit-learn", "lightgbm", "safetensors", "pyarrow",
]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_notebook(bundle):
    source = (bundle / "original.ipynb").read_text()
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Notebook hash mismatch: expected the reviewed v8 source.")
    return json.loads(source)


def cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def literal_assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    raise ValueError(f"Missing literal assignment: {name}")


def prepare(bundle, response_path=None):
    if response_path:
        payload = json.loads(response_path.read_text())
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            payload = json.load(response)
    meta = payload["metadata"]
    source = payload["blob"]["source"]
    if meta["currentVersionNumber"] != 8 or digest(source) != SOURCE_SHA256:
        raise ValueError("Upstream changed; refusing to silently reproduce a different version.")
    nb = json.loads(source)
    scripts = [cell_source(c) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, script in enumerate(scripts):
        compile(script, f"original-cell-{i}", "exec")
    embedded = literal_assignment(cell_source(nb["cells"][3]), "_GAUSSIAN_SOURCE")
    compile(embedded, "public_gaussian_stack.py", "exec")
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "original.ipynb").write_text(source)
    (bundle / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Inspection copies only; run() uses the hash-checked notebook.
    (bundle / "original.py").write_text(
        "# Exported from lynnsakurai/llm-decoder-qwen-ver v8; see metadata.json.\n\n"
        + "\n\n# %%\n\n".join(scripts)
    )
    (bundle / "public_gaussian_stack.py").write_text(embedded)
    (bundle / "provenance.json").write_text(json.dumps({
        "source_url": SOURCE_URL,
        "version": 8,
        "source_sha256": SOURCE_SHA256,
        "last_run_time": meta["lastRunTime"],
        "docker_image": meta["dockerImage"],
        "warning": "Attached datasets and notebook outputs are not fully version-pinned by this API.",
    }, indent=2) + "\n")
    print(f"Prepared verified v8 source: {bundle}")


def validate_input_path(path):
    path = path.resolve()
    # Paths are substituted inside the notebook's embedded Python string too.
    if any(c in str(path) for c in ("'", '"', "\\", "\n", "\r")):
        raise ValueError("Input path cannot contain quotes, backslashes or newlines.")
    return path


def remap_source(source, input_root):
    return source.replace("/kaggle/input", str(validate_input_path(input_root)))


def asset_report(nb, input_root):
    preflight = cell_source(nb["cells"][2])
    required = literal_assignment(preflight, "_required")
    lora_files = literal_assignment(preflight, "_lora24_files")
    items = {}
    for label, pattern in required.items():
        hits = sorted(str(p) for p in input_root.glob(pattern))
        items[label] = {"ok": bool(hits), "matches": hits}
    loras = sorted(p for p in input_root.glob("**/lora24/final") if p.is_dir())
    valid = [str(p) for p in loras if all((p / f).is_file() for f in lora_files)]
    items["original LoRA24"] = {
        "ok": len(valid) == 1, "matches": valid,
        "incomplete": {str(p): [f for f in lora_files if not (p / f).is_file()]
                       for p in loras if str(p) not in valid},
    }
    contests = [
        input_root / "competitions/name-that-llm", input_root / "name-that-llm",
    ]
    items["competition CSVs"] = {
        "ok": any(all((p / f).is_file() for f in ("train.csv", "test.csv"))
                  for p in contests),
        "matches": [str(p) for p in contests
                    if all((p / f).is_file() for f in ("train.csv", "test.csv"))],
    }
    configs = []
    for path in input_root.glob("**/config.json"):
        try:
            configs.append((path, json.loads(path.read_text())))
        except (OSError, ValueError):
            pass
    for label, (kind, hidden) in BASES.items():
        matches = [str(p.parent) for p, c in configs
                   if c.get("model_type") == kind
                   and (hidden is None or c.get("hidden_size") == hidden)]
        items[label] = {"ok": bool(matches), "matches": matches}
    ultra = []
    for root in input_root.glob("**/ultrabert-380m-ntl"):
        folds = [p for p, c in configs
                 if p.parent.parent == root and p.parent.name.startswith("fold")
                 and c.get("model_type") == "ultrabert"]
        if len(folds) == 5:
            ultra.append(str(root))
    items["UltraBERT five folds"] = {"ok": bool(ultra), "matches": ultra}
    atlas = items["ATLAS bundle"]
    atlas["ok"] = len(atlas["matches"]) == 1
    return items


def environment_report():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    env = {"python": sys.version, "executable": sys.executable, "packages": packages}
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["gpus"] = [torch.cuda.get_device_name(i)
                       for i in range(torch.cuda.device_count())]
    except Exception as exc:
        env["cuda_available"] = False
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        env["model_types"] = {
            k: k in CONFIG_MAPPING_NAMES
            for k in ("qwen3", "qwen2", "glm4", "qwen3_5", "ultrabert")
        }
    except Exception as exc:
        env["transformers_error"] = f"{type(exc).__name__}: {exc}"
        env["model_types"] = {}
    return env


def check(bundle, input_root, output):
    nb = load_notebook(bundle)
    assets = asset_report(nb, input_root)
    env = environment_report()
    # Mirrors the notebook's native model-type requirements. This is a
    # preliminary check, not a guarantee of weight or numerical compatibility.
    ready = (all(v["ok"] for v in assets.values())
             and all(env["packages"].values())
             and env["cuda_available"]
             and all(env["model_types"].get(k, False)
                     for k in ("qwen3", "qwen2", "glm4", "qwen3_5", "ultrabert")))
    report = {"preliminary_ready": bool(ready), "input_root": str(input_root),
              "assets": assets, "environment": env}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for label, item in assets.items():
        print(f"{'OK     ' if item['ok'] else 'MISSING'} {label}")
    print("Packages:", json.dumps(env["packages"]))
    print("CUDA:", env["cuda_available"], "model types:", env["model_types"])
    print(f"Report: {output}\nPreliminary ready: {ready}")
    return ready


def run(bundle, input_root, output):
    if os.environ.get("NTL_SMOKE") == "1" or os.environ.get("NTL_TINY") == "1":
        raise ValueError("Unset NTL_SMOKE and NTL_TINY for original inference.")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "submission.parquet").exists():
        raise FileExistsError("submission.parquet already exists; use a new output directory.")
    if not check(bundle, input_root, output / "preflight.json"):
        raise RuntimeError("Missing inputs/environment; inference was not started.")
    nb = load_notebook(bundle)
    # Keep numerical code, model ordering, weights, grouping and balancing intact.
    # Only the filesystem root and working directory differ from Kaggle.
    namespace = {"__name__": "__main__"}
    cwd = Path.cwd()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        os.chdir(output)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                print(f"\n=== Original notebook cell {i} ===", flush=True)
                source = remap_source(cell_source(cell), input_root)
                exec(compile(source, f"decoder-v8-cell-{i}", "exec"), namespace)
    finally:
        os.chdir(cwd)
    print(f"Finished: {output / 'submission.parquet'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download and hash-check source (not model weights)")
    p.add_argument("--api-response", type=Path, help="Use a previously downloaded API JSON")
    for p in [p, sub.add_parser("check"), sub.add_parser("run")]:
        p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
        if p.prog.endswith(("check", "run")):
            p.add_argument("--input-root", type=Path,
                           default=DEFAULT_BUNDLE / "input")
            p.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.command == "prepare":
        prepare(bundle, args.api_response)
    else:
        inputs = validate_input_path(args.input_root)
        if args.command == "check":
            report = (args.output or bundle / "preflight.json").resolve()
            return 0 if check(bundle, inputs, report) else 2
        run(bundle, inputs, (args.output or bundle / "run").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
