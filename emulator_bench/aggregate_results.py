"""Mean and std over seeds of the EC level accuracy (k=1, preliminary ECs dropped) for every Theia split."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import DEFAULT_OUT_ROOT, TASK, atomic_write, load_json  # noqa: E402


def aggregate(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted((Path(out_root) / TASK).glob("*/*/seed_*/metrics.json")):
        metrics = load_json(path)
        accuracy = metrics["ec_level_accuracy_no_b"]
        rows.append({"split_group": path.parents[2].name, "split_name": path.parents[1].name, "seed": metrics["seed"],
                     "n_queries": accuracy["n_queries"], **accuracy["k=1"]})
    runs = pd.DataFrame(rows)
    levels = [column for column in runs.columns if column.startswith("level_")]
    summary = runs.groupby(["split_group", "split_name"])[levels + ["n_queries"]].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    return runs, summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Theia runs over seeds.")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()
    runs, summary = aggregate(Path(args.out_root))
    atomic_write(Path(args.out_root) / TASK / "summary_runs.csv", lambda tmp: runs.to_csv(tmp, index=False))
    atomic_write(Path(args.out_root) / TASK / "summary_by_split.csv", lambda tmp: summary.to_csv(tmp, index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
