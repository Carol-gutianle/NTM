"""Run Kaggle CLI with locally stored credentials, without printing secrets."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/ntl-kaggle-client")
for line in (root / ".env").read_text().splitlines():
    if line.startswith("KAGGLE_API_TOKEN="):
        os.environ["KAGGLE_API_TOKEN"] = line.split("=", 1)[1].strip()
os.environ["KAGGLE_CONFIG_DIR"] = str(root / "artifacts/kaggle-config")
Path(os.environ["KAGGLE_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
from kaggle.cli import main
main()
