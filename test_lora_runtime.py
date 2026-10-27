"""CPU-only tests for cooperative checkpoint flags and GPU telemetry parsing."""
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lora_runtime import RuntimeGuard


class RuntimeTests(unittest.TestCase):
    def test_stop_and_checkpoint_marker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            guard = RuntimeGuard(root, enabled=False)
            args = SimpleNamespace(device="cpu", eval_strategy="epoch", use_cpu=True)
            state = SimpleNamespace(global_step=2, max_steps=10, is_world_process_zero=True)
            control = SimpleNamespace(should_save=False, should_evaluate=False,
                                      should_training_stop=False)
            (root / "STOP_REQUEST").touch()
            guard.on_step_end(args, state, control)
            self.assertTrue(control.should_save and control.should_evaluate)
            self.assertTrue(control.should_training_stop and guard.interrupted)
            (root / "checkpoint-2").mkdir()
            guard.on_save(args, state, control)
            self.assertTrue((root / "checkpoint-2/COMPLETE").is_file())

    def test_signal_and_full_training(self):
        with TemporaryDirectory() as directory:
            guard = RuntimeGuard(directory, enabled=False)
            args = SimpleNamespace(device="cpu", eval_strategy="no", use_cpu=True)
            state = SimpleNamespace(global_step=1, max_steps=10, is_world_process_zero=True)
            control = SimpleNamespace(should_save=False, should_evaluate=False,
                                      should_training_stop=False)
            guard._signal(signal.SIGTERM, None)
            guard.on_step_end(args, state, control)
            self.assertTrue(control.should_save and control.should_training_stop)
            self.assertFalse(control.should_evaluate)

    def test_real_telemetry_requests_stop(self):
        with TemporaryDirectory() as directory:
            guard = RuntimeGuard(directory, interval=.001, grace=0, patience=0,
                                 threshold=20, action="stop")
            guard.phase = "train"
            sample = SimpleNamespace(stdout="0, GPU-example, 0, 1024, 81920\n")
            def sample_and_finish(*args, **kwargs):
                guard.finished.set()
                return sample
            with patch("lora_runtime.subprocess.run", side_effect=sample_and_finish):
                guard._monitor()
            record = json.loads((Path(directory) / "gpu_metrics.jsonl").read_text())
            self.assertEqual(record["event"], "sustained_low_utilization")
            self.assertTrue(guard.request_save.is_set() and guard.request_stop.is_set())

    def test_final_step_saves_even_before_interval(self):
        with TemporaryDirectory() as directory:
            guard = RuntimeGuard(directory, enabled=False)
            args = SimpleNamespace(device="cpu", eval_strategy="steps", use_cpu=True)
            state = SimpleNamespace(global_step=3, max_steps=3, is_world_process_zero=True)
            control = SimpleNamespace(should_save=False, should_evaluate=False,
                                      should_training_stop=False)
            guard.on_step_end(args, state, control)
            self.assertTrue(control.should_save and control.should_evaluate)
            self.assertFalse(guard.interrupted)
            # A checkpoint rotated away during replay must not recreate an empty directory.
            guard.on_save(args, state, control)
            self.assertFalse((Path(directory) / "checkpoint-3").exists())


if __name__ == "__main__":
    unittest.main()
