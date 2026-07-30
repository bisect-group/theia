import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from emulator_bench.aggregate_results import aggregate, discover_runs, inspect_record, main
from emulator_bench.ec import build_level_vocab, hierarchy_accuracies


class AggregateResultsTest(unittest.TestCase):
    def write_run(self, base, split, level, seed, state="valid"):
        directory = base / split / "theia_results" / level / f"seed_{seed}"
        directory.mkdir(parents=True)
        frame = pd.DataFrame({"ec_number": ["1.1.1.1", "2.1.1.1"], "label": ["1", "2"], "prediction": ["1", "1"]})
        frame.to_csv(directory / "predictions.csv", index=False)
        if state == "malformed":
            (directory / "metrics.json").write_text("{")
        elif state != "missing":
            from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
            metrics = {"level": level, "loss": 1.0, "accuracy": float(accuracy_score(frame.label, frame.prediction)), "macro_f1": float(f1_score(frame.label, frame.prediction, average="macro", zero_division=0)), "weighted_f1": float(f1_score(frame.label, frame.prediction, average="weighted", zero_division=0)), "mcc": float(matthews_corrcoef(frame.label, frame.prediction)), "n_samples": len(frame)}
            vocab = {name: build_level_vocab(frame.ec_number, name) for name in ("ec1", "ec12", "ec123", "ec1234")}
            metrics.update(hierarchy_accuracies(frame.ec_number.tolist(), frame.prediction.tolist(), vocab_by_level=vocab))
            if state == "mismatch":
                metrics["accuracy"] = 0.0
            (directory / "metrics.json").write_text(json.dumps(metrics))

    def test_discovers_nested_split_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.write_run(base, "random_splits", "ec1", 42)
            self.write_run(base, "uniprot_time_splits", "ec12", 43)
            self.write_run(base, "reaction_drfp_tanimoto_splits/threshold_0.15", "ec123", 44)
            records = [inspect_record(record) for record in discover_runs(base, "theia_results", None)]
            self.assertEqual(len(records), 3)
            summaries, _ = aggregate(records)
            self.assertEqual(sum(row["runs_completed"] for row in summaries), 3)
            self.assertIn("reaction_drfp_tanimoto_splits/threshold_0.15", {row["split_name"] for row in summaries})

    def test_marks_malformed_and_mismatched_runs_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.write_run(base, "random_splits", "ec1", 42)
            self.write_run(base, "random_splits", "ec1", 43, "malformed")
            self.write_run(base, "random_splits", "ec1", 44, "mismatch")
            records = [inspect_record(record) for record in discover_runs(base, "theia_results", None)]
            self.assertEqual({record["seed"]: record["status"] for record in records}, {42: "valid", 43: "invalid", 44: "invalid"})

    def test_strict_mode_writes_diagnostics_then_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, output = Path(tmp) / "base", Path(tmp) / "out"
            self.write_run(base, "random_splits", "ec1", 42, "missing")
            with patch("sys.argv", ["aggregate_results", "--base-dir", str(base), "--output-dir", str(output), "--strict"]):
                with self.assertRaises(SystemExit):
                    main()
            self.assertTrue((output / "diagnostics.json").is_file())


if __name__ == "__main__":
    unittest.main()
