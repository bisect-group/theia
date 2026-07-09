import argparse
import json
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.cache_features import cache_job
from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_DEVICE, DEFAULT_EMBEDDINGS_DIR, cached_split_path, console, discover_split_jobs, save_json
from emulator_bench.train import train_one


def objective(trial, job, level: str, args):
    hparams = {
        "input_dim": 10240,
        "hidden_dim": 1664,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "epochs": args.epochs,
        "patience": trial.suggest_int("patience", 3, 8),
        "min_delta": trial.suggest_float("min_delta", 1e-4, 2e-3, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "scheduler_gamma": trial.suggest_float("scheduler_gamma", 0.85, 0.98),
    }
    trial_dir = Path(args.study_dir) / f"trial_{trial.number}"
    hparams_path = trial_dir / "hparams.json"
    save_json(hparams_path, hparams)
    train_args = argparse.Namespace(
        train_path=str(cached_split_path(job, level, "train")),
        val_path=str(cached_split_path(job, level, "val")),
        test_path=str(cached_split_path(job, level, "test")),
        out_dir=str(trial_dir),
        level=level,
        hparams_json=str(hparams_path),
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        overwrite=False,
    )
    metrics = train_one(train_args)
    return metrics["accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune retraining-safe Theia hyperparameters with Optuna.")
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--split_group", default="random_splits")
    parser.add_argument("--split_name", default="random_splits")
    parser.add_argument("--level", default="ec1234")
    parser.add_argument("--study_dir", default="")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=[args.split_group])
    job = next((candidate for candidate in jobs if candidate.split_name == args.split_name), None)
    if job is None:
        raise FileNotFoundError(f"Split job not found: {args.split_group}/{args.split_name}")
    cache_job(job, [args.level], Path(args.embeddings_dir), max_rows=args.max_rows)
    if not args.study_dir:
        args.study_dir = str(job.root_dir / "optuna_studies" / args.level)
    Path(args.study_dir).mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{Path(args.study_dir) / 'study.db'}"
    study = optuna.create_study(direction="maximize", study_name=f"theia_{args.level}", storage=storage, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, job, args.level, args), n_trials=args.n_trials)
    best_path = Path(args.study_dir) / f"theia_{args.level}_optuna_best_hparams.json"
    save_json(best_path, {"best_hparams": study.best_params, "best_value": study.best_value})
    console.print(f"Wrote {best_path}")


if __name__ == "__main__":
    main()
