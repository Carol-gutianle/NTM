"""Cooperative checkpointing and real GPU telemetry; never generates GPU load."""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import torch
from transformers import TrainerCallback
from transformers import Trainer


class CheckpointTrainer(Trainer):
    def _save_checkpoint(self, model, trial):
        # Resuming an earlier step can overwrite a previously complete directory.
        # Invalidate its marker BEFORE any files change.
        if self.is_world_process_zero():
            marker = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}" / "COMPLETE"
            marker.unlink(missing_ok=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        super()._save_checkpoint(model, trial)


class RuntimeGuard(TrainerCallback):
    def __init__(self, output, interval=10, threshold=20, patience=180,
                 grace=180, action="warn", enabled=True):
        self.output = Path(output)
        self.interval = interval
        self.threshold = threshold
        self.patience = patience
        self.grace = grace
        self.action = action
        self.enabled = enabled
        self.finished = threading.Event()
        self.request_save = threading.Event()
        self.request_stop = threading.Event()
        self.interrupted = False
        self.thread = None
        self.handlers = {}
        self.phase = "startup"
        self.step = 0

    def _signal(self, signum, frame):
        # No CUDA/distributed work in a signal handler.
        self.request_stop.set()

    def _monitor(self):
        start = time.monotonic()
        low_since = None
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            command += ["--id", visible]
        try:
            with (self.output / "gpu_metrics.jsonl").open("a", buffering=1) as log:
                while not self.finished.is_set():
                    result = subprocess.run(command, capture_output=True, text=True,
                                            check=True, timeout=max(5, self.interval))
                    gpus = []
                    for line in result.stdout.strip().splitlines():
                        index, uuid, util, used, total = [x.strip() for x in line.split(",")]
                        gpus.append(dict(index=index, uuid=uuid, utilization=float(util),
                                         memory_used_mb=float(used), memory_total_mb=float(total)))
                    now = time.monotonic()
                    # Ignore startup, validation and saving: not sustained training load.
                    low = (bool(gpus) and self.phase == "train"
                           and now - start >= self.grace
                           and min(g["utilization"] for g in gpus) < self.threshold)
                    low_since = (low_since if low_since is not None else now) if low else None
                    record = dict(time=time.time(), step=self.step, phase=self.phase, gpus=gpus)
                    if low_since is not None and now - low_since >= self.patience:
                        record["event"] = "sustained_low_utilization"
                        record["action"] = self.action
                        print(f"[GPU guard] 持续低利用率（至少一张卡 < {self.threshold}%）；"
                              f"请求检查点，action={self.action}。请检查真实训练吞吐/平台规则。",
                              flush=True)
                        self.request_save.set()
                        if self.action == "stop":
                            self.request_stop.set()
                        low_since = now
                    log.write(json.dumps(record) + "\n")
                    self.finished.wait(self.interval)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            print(f"[GPU guard] 监控不可用，训练仍继续：{exc}", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal)
        if state.is_world_process_zero and self.enabled and not args.use_cpu:
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()

    def on_step_begin(self, args, state, control, **kwargs):
        self.phase = "train"

    def on_step_end(self, args, state, control, **kwargs):
        self.step = state.global_step
        stop_file = self.output / "STOP_REQUEST"
        stop = self.request_stop.is_set() or (
            state.is_world_process_zero and stop_file.exists())
        flags = torch.tensor([int(self.request_save.is_set() or stop), int(stop)],
                             device=args.device, dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        save, stop = flags.tolist()
        if save or state.global_step >= state.max_steps:
            control.should_save = True
            # Keep best-checkpoint selection valid when adding an unscheduled save.
            control.should_evaluate = args.eval_strategy != "no"
            self.request_save.clear()
        if stop:
            self.interrupted = True
            control.should_training_stop = True
        if control.should_evaluate or control.should_save:
            self.phase = "eval_or_save"
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        self.phase = "eval_or_save"

    def on_save(self, args, state, control, **kwargs):
        # All ranks' optimizer/RNG files must be finished before marking resumable.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if state.is_world_process_zero:
            checkpoint = self.output / f"checkpoint-{state.global_step}"
            # Trainer may immediately rotate this checkpoint when replaying an
            # older step in an output directory containing later checkpoints.
            if checkpoint.is_dir():
                (checkpoint / "COMPLETE").write_text("checkpoint files saved on all ranks\n")
        self.phase = "train"

    def close(self):
        self.finished.set()
        if self.thread:
            self.thread.join(timeout=max(5, self.interval) + 1)
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)

    def on_train_end(self, args, state, control, **kwargs):
        self.close()
