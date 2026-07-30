"""Aggregate completed Theia EMULaToR test evaluations across seeds."""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pandas as pd
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_LEVELS, DEFAULT_RESULTS_DIRNAME, atomic_write_text, console, load_json
from emulator_bench.ec import build_level_vocab, hierarchy_accuracies


METRIC_TOLERANCE = 1e-9
VALIDATED_METRICS = {"accuracy", "macro_f1", "weighted_f1", "mcc", "n_samples", "ec1_accuracy", "ec12_accuracy", "ec123_accuracy", "ec1234_accuracy"}


def csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def numeric_metrics(path: Path) -> dict[str, float]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("metrics.json must contain a non-empty object")
    result = {}
    for key, value in payload.items():
        if key == "level":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"metric {key!r} is not a finite number")
        result[str(key)] = float(value)
    return result


def recompute_metrics(predictions_path: Path) -> dict[str, float]:
    predictions = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)
    missing = {"ec_number", "label", "prediction"} - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing columns: {', '.join(sorted(missing))}")
    true = predictions["label"].tolist()
    predicted = predictions["prediction"].tolist()
    metrics = {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, predicted, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(true, predicted)) if len(set(true)) > 1 and len(set(predicted)) > 1 else 0.0,
        "n_samples": float(len(predictions)),
    }
    vocab = {level: build_level_vocab(predictions["ec_number"].tolist(), level) for level in DEFAULT_LEVELS}
    metrics.update(hierarchy_accuracies(predictions["ec_number"].tolist(), predicted, vocab_by_level=vocab))
    return metrics


def discover_runs(base_dir: Path, results_dirname: str, split_groups: list[str] | None) -> list[dict[str, Any]]:
    records = []
    for results_dir in sorted(path for path in base_dir.rglob(results_dirname) if path.is_dir()):
        split_root = results_dir.parent
        relative = split_root.relative_to(base_dir)
        if not relative.parts:
            continue
        split_group = relative.parts[0]
        if split_groups and split_group not in split_groups:
            continue
        for level_dir in sorted(path for path in results_dir.iterdir() if path.is_dir() and path.name in DEFAULT_LEVELS):
            for seed_dir in sorted(path for path in level_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
                seed_text = seed_dir.name.removeprefix("seed_")
                try:
                    seed: int | str = int(seed_text)
                except ValueError:
                    seed = seed_text
                records.append({"split_group": split_group, "split_name": relative.as_posix(), "level": level_dir.name, "seed": seed, "run_dir": seed_dir})
    return records


def inspect_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    metrics_path = result["run_dir"] / "metrics.json"
    predictions_path = result["run_dir"] / "predictions.csv"
    result.update(run_dir=str(result["run_dir"]), status="valid", error="", metrics={})
    missing = [path.name for path in (metrics_path, predictions_path) if not path.is_file()]
    if missing:
        result.update(status="missing", error=f"missing required artifact(s): {', '.join(missing)}")
        return result
    try:
        payload = load_json(metrics_path)
        if payload.get("level") != result["level"]:
            raise ValueError(f"metric level {payload.get('level')!r} does not match directory {result['level']!r}")
        metrics = numeric_metrics(metrics_path)
        recomputed = recompute_metrics(predictions_path)
        mismatches = [
            f"{name}: saved={metrics[name]:.12g}, recomputed={recomputed[name]:.12g}"
            for name in sorted(VALIDATED_METRICS & set(metrics) & set(recomputed))
            if not math.isclose(metrics[name], recomputed[name], rel_tol=METRIC_TOLERANCE, abs_tol=METRIC_TOLERANCE)
        ]
        result["metrics"] = metrics
        if mismatches:
            result.update(status="invalid", error="; ".join(mismatches))
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        result.update(status="invalid", error=str(error))
    return result


def aggregate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["split_group"], record["split_name"], record["level"])].append(record)
    rows, payload = [], []
    for identity, members in sorted(groups.items()):
        valid = [member for member in members if member["status"] == "valid"]
        row: dict[str, Any] = {
            "split_group": identity[0], "split_name": identity[1], "level": identity[2],
            "runs_discovered": len(members), "runs_completed": len(valid),
            "runs_missing": sum(member["status"] == "missing" for member in members),
            "runs_invalid": sum(member["status"] == "invalid" for member in members),
            "completed_seeds": ",".join(str(member["seed"]) for member in valid),
        }
        metrics_json = {}
        for name in sorted({name for member in valid for name in member["metrics"] if name != "n_samples"}):
            values = [member["metrics"][name] for member in valid if name in member["metrics"]]
            metrics_json[name] = {"count": len(values), "mean": mean(values) if values else None, "sample_std": stdev(values) if len(values) > 1 else 0.0 if values else None}
            row[f"{name}_mean"] = metrics_json[name]["mean"]
            row[f"{name}_sample_std"] = metrics_json[name]["sample_std"]
        rows.append(row)
        fixed = {key: value for key, value in row.items() if not key.endswith("_mean") and not key.endswith("_sample_std")}
        payload.append({**fixed, "metrics": metrics_json})
    return rows, payload


def write_reports(output_dir: Path, records: list[dict[str, Any]], summary_rows: list[dict[str, Any]], summary_json: list[dict[str, Any]]) -> None:
    metric_names = sorted({name for record in records for name in record["metrics"]})
    per_run_rows = [{key: value for key, value in record.items() if key != "metrics"} | record["metrics"] for record in records]
    run_fields = ["split_group", "split_name", "level", "seed", "run_dir", "status", "error"] + metric_names
    summary_metrics = sorted({key for row in summary_rows for key in row if key.endswith("_mean") or key.endswith("_sample_std")})
    summary_fields = ["split_group", "split_name", "level", "runs_discovered", "runs_completed", "runs_missing", "runs_invalid", "completed_seeds"] + summary_metrics
    diagnostics = [{key: value for key, value in record.items() if key != "metrics"} for record in records if record["status"] != "valid"]
    atomic_write_text(output_dir / "per_run_metrics.csv", csv_text(per_run_rows, run_fields))
    atomic_write_text(output_dir / "summary_metrics.csv", csv_text(summary_rows, summary_fields))
    atomic_write_text(output_dir / "summary_metrics.json", __import__("json").dumps({"groups": summary_json}, indent=2, sort_keys=True) + "\n")
    atomic_write_text(output_dir / "diagnostics.json", __import__("json").dumps({"runs": diagnostics}, indent=2, sort_keys=True) + "\n")


def render_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    table = Table(title="Theia aggregated metrics")
    for column in ("split", "level", "completed", "missing", "invalid"):
        table.add_column(column)
    for row in rows:
        table.add_row(row["split_name"], row["level"], str(row["runs_completed"]), str(row["runs_missing"]), str(row["runs_invalid"]))
    console.print(table)
    console.print(f"Reports written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Theia benchmark metrics across seeds.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--results-dirname", default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-groups", nargs="*")
    parser.add_argument("--strict", action="store_true", help="Fail after writing reports if any discovered run is missing or invalid.")
    args = parser.parse_args()
    output_dir = args.output_dir or args.base_dir / "aggregated_metrics"
    records = [inspect_record(record) for record in discover_runs(args.base_dir, args.results_dirname, args.split_groups)]
    if not records:
        parser.error(f"No {args.results_dirname!r} seed runs found under {args.base_dir}")
    summary_rows, summary_json = aggregate(records)
    write_reports(output_dir, records, summary_rows, summary_json)
    render_summary(summary_rows, output_dir)
    if args.strict and any(record["status"] != "valid" for record in records):
        raise SystemExit("Strict aggregation failed: one or more runs are missing or invalid; see diagnostics.json.")


if __name__ == "__main__":
    main()
