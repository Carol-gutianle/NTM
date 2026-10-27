"""Wait for bulk generation to release its lock, snapshot candidates, then evaluate.

No API calls or retraining of neural models. An incomplete generation run is
explicitly labeled partial; results never imply every class reached 200.
"""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
run = ROOT / "artifacts/augmentation200"
out = ROOT / "artifacts/augmentation200_evaluation"
status_path = run / "evaluation_status.json"


def status(**fields):
    value = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **fields}
    temp = status_path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(status_path)
    print(json.dumps(value), flush=True)


if out.exists():
    raise SystemExit("Evaluation output already exists; inspect before repeating")
status(state="waiting_for_generation")
with (run / "run.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    from generate_augmentation200 import read_journal, summarize
    manifest = json.loads((run / "bulk_manifest.json").read_text())
    progress = summarize(run, manifest["target_labels"], manifest["prompts"],
                         read_journal(run / "generation_journal.jsonl"))
    snapshot = run / "augmentation_evaluation_snapshot.csv"
    shutil.copyfile(run / "augmentation_candidates.csv", snapshot)
if not progress["accepted_rows"]:
    status(state="no_candidates", progress=progress)
    raise SystemExit(1)
status(state="evaluating", progress=progress)
try:
    with (run / "evaluation.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-u", str(ROOT / "evaluate_augmentation.py"),
             "--augmentation", str(snapshot), "--output", str(out)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    if result.returncode:
        status(state="evaluation_failed", returncode=result.returncode,
               log=str(run / "evaluation.log"))
        raise SystemExit(result.returncode)
    summary = json.loads((out / "summary.json").read_text())
    status(state="evaluation_complete",
           generation_all_18_classes_complete=progress["all_available_classes_complete"],
           generation_progress=progress,
           original_accuracy=summary["results"]["original"]["accuracy"],
           augmented_accuracy=summary["results"]["augmented"]["accuracy"],
           accuracy_delta=summary["accuracy_delta"],
           summary_path=str(out / "summary.json"))
except subprocess.TimeoutExpired:
    status(state="evaluation_timeout", log=str(run / "evaluation.log"))
    raise SystemExit(1)
