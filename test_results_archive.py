"""Offline, standard-library-only checks of the published experiment evidence."""
import json
from pathlib import Path
import unittest

from export_results import RUNS, calculate_metrics, csv_rows, digest

RESULTS = Path(__file__).resolve().parent / "results"
EVIDENCE = RESULTS / "evidence"


class ResultsArchiveTests(unittest.TestCase):
    def test_manifest_integrity_and_allowlist(self):
        manifest = json.loads((RESULTS / "manifest.json").read_text())
        listed = set()
        for item in manifest["files"]:
            path = RESULTS / item["destination"]
            self.assertTrue(path.resolve().is_relative_to(EVIDENCE.resolve()))
            self.assertIn(path.suffix, {".json", ".csv"})
            data = path.read_bytes()
            self.assertEqual(digest(data), item["export_sha256"])
            self.assertEqual(len(data), item["bytes"])
            if path.suffix == ".csv":
                rows = csv_rows(path)
                self.assertFalse(rows and "text" in rows[0])
            listed.add(path)
        self.assertEqual(listed, {p for p in EVIDENCE.rglob("*") if p.is_file()})

    def test_holdout_metrics_and_alignment(self):
        split = csv_rows(EVIDENCE / "split.csv")
        expected = {r["id"]: r for r in split if r["split"] == "validation"}
        self.assertEqual(len(expected), 540)
        self.assertEqual(len(split), 2699)
        training_prompts = {r["prompt_id"] for r in split if r["split"] == "train"}
        validation_prompts = {r["prompt_id"] for r in expected.values()}
        self.assertEqual((len(training_prompts), len(validation_prompts)), (120, 30))
        self.assertFalse(training_prompts & validation_prompts)
        summary = json.loads((RESULTS / "summary.json").read_text())
        self.assertEqual(len(summary), len(RUNS))
        for run in summary:
            rows = csv_rows(RESULTS / run["evidence"])
            self.assertEqual({r["id"] for r in rows}, set(expected))
            for row in rows:
                self.assertEqual(row["model"], expected[row["id"]]["model"])
                self.assertEqual(row["prompt_id"], expected[row["id"]]["prompt_id"])
            for key, value in calculate_metrics(rows).items():
                self.assertAlmostEqual(run[key], value)

    def test_grouped_cv_and_pooled_oof(self):
        directory = EVIDENCE / "tfidf_char_cv"
        summary = json.loads((directory / "summary.json").read_text())
        assignments = {r["id"]: r for r in csv_rows(directory / "fold_assignments.csv")}
        all_ids, prompts_by_fold, predictions = set(), [], {}
        for fold in summary["folds"]:
            rows = csv_rows(directory / f"fold_{fold['fold']}" / "validation_predictions.csv")
            metrics = calculate_metrics(rows)
            self.assertAlmostEqual(metrics["balanced_accuracy"], fold["balanced_accuracy"])
            self.assertEqual(len(rows), fold["validation_rows"])
            ids = {r["id"] for r in rows}
            self.assertFalse(ids & all_ids)
            all_ids |= ids
            prompts = {r["prompt_id"] for r in rows}
            self.assertEqual(len(prompts), 30)
            self.assertTrue(all(not prompts & previous for previous in prompts_by_fold))
            prompts_by_fold.append(prompts)
            for row in rows:
                self.assertEqual(int(assignments[row["id"]]["validation_fold"]), fold["fold"])
                self.assertEqual(assignments[row["id"]]["prompt_id"], row["prompt_id"])
                self.assertEqual(assignments[row["id"]]["model"], row["model"])
                predictions[row["id"]] = row
        self.assertEqual(len(all_ids), 2699)
        oof = csv_rows(directory / "oof/validation_predictions.csv")
        self.assertEqual({r["id"]: r for r in oof}, predictions)
        self.assertAlmostEqual(calculate_metrics(oof)["balanced_accuracy"],
                               summary["pooled_oof_balanced_accuracy"])

    def test_augmentation_completion_and_delta(self):
        progress = json.loads((EVIDENCE / "augmentation200/progress.json").read_text())
        summary = json.loads((EVIDENCE / "augmentation200_evaluation/summary.json").read_text())
        self.assertFalse(progress["all_available_classes_complete"])
        self.assertEqual(sum(progress["actual_per_class"].values()), 2025)
        self.assertEqual(sum(v == 200 for v in progress["actual_per_class"].values()), 8)
        self.assertEqual(summary["augmentation_rows_used"], 2025)
        self.assertEqual(len(summary["missing_augmented_classes"]), 6)
        results = summary["results"]
        self.assertAlmostEqual(results["augmented"]["accuracy"] - results["original"]["accuracy"],
                               summary["accuracy_delta"])
        self.assertEqual(summary["fixed_wrong_to_right"] - summary["regressed_right_to_wrong"],
                         round(summary["accuracy_delta"] * 540))


if __name__ == "__main__":
    unittest.main()
