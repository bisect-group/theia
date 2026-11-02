import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emulator_bench.aggregate_results import aggregate  # noqa: E402


def write_run(out_root: Path, seed: int, level_4: float) -> None:
    run = out_root / "enzyme_retrieval" / "uniprot_time_splits" / "all" / f"seed_{seed}"
    run.mkdir(parents=True)
    accuracy = {"k=1": {f"level_{n}_accuracy": level_4 + (4 - n) for n in (1, 2, 3, 4)}, "n_queries": 10, "n_rows": 12}
    (run / "metrics.json").write_text(json.dumps({"seed": seed, "ec_level_accuracy_no_b": accuracy}))


def test_aggregate_means_and_stds_over_seeds(tmp_path):
    for seed, level_4 in ((666, 10.0), (777, 20.0), (888, 30.0)):
        write_run(tmp_path, seed, level_4)
    runs, summary = aggregate(tmp_path)
    assert len(runs) == 3
    row = summary.iloc[0]
    assert row["split_group"] == "uniprot_time_splits" and row["split_name"] == "all"
    assert row["level_4_accuracy_mean"] == pytest.approx(20.0)
    assert row["level_4_accuracy_std"] == pytest.approx(10.0)
    assert row["level_1_accuracy_mean"] == pytest.approx(23.0)
