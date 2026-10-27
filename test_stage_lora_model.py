"""Small local fixtures only: never read/copy the real 32B weights."""
from contextlib import redirect_stderr
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from stage_lora_model import inventory, stage


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "model"
        self.source.mkdir()
        (self.source / "config.json").write_text('{"model_type":"qwen3"}')
        (self.source / "tokenizer.json").write_text("{}")
        (self.source / "a.safetensors").write_bytes(b"weight-a" * 512)
        (self.source / "b.safetensors").write_bytes(b"weight-b" * 512)
        (self.source / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
        (self.source / ".env").write_text("must not be copied")
        self.cache = self.root / "cache"
        self.addCleanup(patch.stopall)
        patch("stage_lora_model.filesystem", return_value="ext4").start()
        self.log = io.StringIO()

    def copy(self, **kwargs):
        with redirect_stderr(self.log):
            return stage(self.source, self.cache, reserve_bytes=0, **kwargs)

    def test_copy_reuse_and_no_secrets(self):
        target = self.copy()
        self.assertFalse((target / ".env").exists())
        self.assertTrue((target / ".stage-complete.json").is_file())
        before = (target / "a.safetensors").stat().st_mtime_ns
        self.assertEqual(self.copy(), target)
        self.assertEqual((target / "a.safetensors").stat().st_mtime_ns, before)
        self.assertIn("Reusing complete cache", self.log.getvalue())

    def test_resume_missing_file_without_recopying_good_shard(self):
        target = self.copy()
        before = (target / "a.safetensors").stat().st_mtime_ns
        (target / ".stage-complete.json").unlink()
        (target / "b.safetensors").unlink()
        (target / "b.safetensors.stage-part").write_bytes(b"incomplete")
        self.copy()
        self.assertEqual((target / "a.safetensors").stat().st_mtime_ns, before)
        self.assertEqual((target / "b.safetensors").read_bytes(),
                         (self.source / "b.safetensors").read_bytes())
        self.assertFalse((target / "b.safetensors.stage-part").exists())

    def test_full_hash_verification_repairs_same_size_corruption(self):
        target = self.copy()
        file = target / "a.safetensors"
        stat = file.stat()
        file.write_bytes(b"x" * stat.st_size)
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.copy(verify=True)
        self.assertEqual(file.read_bytes(), (self.source / file.name).read_bytes())

    def test_source_change_creates_new_cache(self):
        old = self.copy()
        (self.source / "a.safetensors").write_bytes(b"new model")
        new = self.copy()
        self.assertNotEqual(new, old)
        self.assertTrue(old.exists())

    def test_reject_shared_or_unknown_mount(self):
        for fs in ["fuse.dop-fuse", "nfs", "tmpfs", "overlay", "unknown"]:
            with patch("stage_lora_model.filesystem", return_value=fs):
                with self.assertRaisesRegex(ValueError, "not verified local disk"):
                    self.copy()
        self.assertFalse(self.cache.exists())

    def test_insufficient_space_copies_nothing(self):
        with patch("stage_lora_model.shutil.disk_usage") as usage:
            usage.return_value.free = 1
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                self.copy()
        self.assertFalse(list(self.cache.rglob("*.safetensors")))

    def test_reject_missing_or_unsafe_shards(self):
        index = self.source / "model.safetensors.index.json"
        for name in ["../outside", "/tmp/outside", "missing.safetensors"]:
            index.write_text(json.dumps({"weight_map": {"x": name}}))
            with self.assertRaises(ValueError):
                inventory(self.source)

    def test_concurrent_cli_uses_one_complete_cache(self):
        command = [sys.executable, str(Path(__file__).with_name("stage_lora_model.py")),
                   "--source", str(self.source), "--cache-root", str(self.cache),
                   "--reserve-gib", "0", "--allow-unverified-fs"]
        first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        second = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output1, error1 = first.communicate(timeout=30)
        output2, error2 = second.communicate(timeout=30)
        self.assertEqual(first.returncode, 0, error1)
        self.assertEqual(second.returncode, 0, error2)
        self.assertEqual(output1.strip(), output2.strip())
        self.assertEqual((error1 + error2).count("Copied a.safetensors"), 1)
        self.assertTrue((Path(output1.strip()) / ".stage-complete.json").exists())

    def test_launcher_stage_only_no_gpu_or_torch(self):
        env = {**os.environ, "MODEL": str(self.source), "PYTHON": sys.executable,
               "LOCAL_MODEL_CACHE": str(self.cache), "LOCAL_CACHE_ALLOW_UNVERIFIED": "1",
               "CACHE_RESERVE_GIB": "0", "STAGE_ONLY": "1", "RESUME": "",
               "OUTPUT": str(self.root / "unused-training-output")}
        result = subprocess.run(
            ["bash", str(Path(__file__).with_name("run_lora_8gpu.sh"))],
            env=env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.cache.glob("*/.stage-complete.json"))), 1)
        self.assertFalse((self.root / "unused-training-output").exists())


if __name__ == "__main__":
    unittest.main()
