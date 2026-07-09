import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.cache_features import cache_job
from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_LEVELS,
    DEFAULT_RESULTS_DIRNAME,
    cached_split_path,
    completed_run,
    console,
    copy_dataset_splits,
    discover_split_jobs,
    render_jobs_table,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train.py"


def train_command(job, level: str, seed: int, args, device: str) -> list:
    out_dir = job.root_dir / args.results_dirname / level / f"seed_{seed}"
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_path",
        str(cached_split_path(job, level, "train")),
        "--val_path",
        str(cached_split_path(job, level, "val")),
        "--test_path",
        str(cached_split_path(job, level, "test")),
        "--out_dir",
        str(out_dir),
        "--level",
        level,
        "--hparams_json",
        args.hparams_json,
        "--device",
        device,
        "--seed",
        str(seed),
        "--num_workers",
        str(args.num_workers),
    ]


def worker(work: queue.Queue, args, failures: list) -> None:
    while True:
        item = work.get()
        if item is None:
            work.task_done()
            return
        job, level, seed, physical_gpu = item
        out_dir = job.root_dir / args.results_dirname / level / f"seed_{seed}"
        try:
            if completed_run(out_dir) and not args.overwrite:
                console.print(f"Skipping complete run {out_dir}")
            else:
                if physical_gpu is None:
                    device = args.device
                    env = os.environ.copy()
                    slot = args.device
                else:
                    device = "cuda:0"
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
                    slot = f"physical GPU {physical_gpu}"
                cmd = train_command(job, level, seed, args, device)
                if args.overwrite:
                    cmd.append("--overwrite")
                console.print(f"Starting {job.split_group}/{job.split_name} {level} seed {seed} on {slot}")
                subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
        except Exception as exc:
            failures.append((job, level, seed, str(exc)))
        finally:
            work.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Theia EMULaToR benchmarks across split jobs.")
    parser.add_argument("--source_dataset_dir", default="")
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--split_groups", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--gpus", nargs="*", default=None, help="Physical GPU IDs to use, for example: --gpus 0 1 2 3")
    parser.add_argument("--runs_per_gpu", type=int, default=1, help="Number of training subprocesses to run concurrently on each GPU.")
    parser.add_argument("--results_dirname", default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--hparams_json", default=str(REPO_ROOT / "emulator_bench" / "default_hparams_theia_original.json"))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_num_workers", type=int, default=1)
    parser.add_argument("--cache_batch_size", type=int, default=1024)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--copy_splits", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser()
    embeddings_dir = Path(args.embeddings_dir).expanduser()
    if args.copy_splits:
        if not args.source_dataset_dir:
            raise ValueError("--source_dataset_dir is required with --copy_splits")
        copied = copy_dataset_splits(Path(args.source_dataset_dir), base_dir, overwrite=args.overwrite)
        console.print(f"Copied {len(copied)} files")
    jobs = discover_split_jobs(base_dir, split_groups=args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found under {base_dir}")
    render_jobs_table(jobs)
    if not args.skip_cache:
        for job in jobs:
            cache_job(
                job,
                args.levels,
                embeddings_dir,
                overwrite=args.overwrite,
                max_rows=args.max_rows,
                batch_size=args.cache_batch_size,
                num_workers=args.cache_num_workers,
            )

    if args.runs_per_gpu < 1:
        raise ValueError("--runs_per_gpu must be >= 1")
    slots = []
    if args.gpus:
        for gpu in args.gpus:
            slots.extend([str(gpu)] * args.runs_per_gpu)
    else:
        slots = [None]
    console.print(f"Parallel training slots: {len(slots)}")
    work: queue.Queue = queue.Queue()
    failures = []
    for index, experiment in enumerate((job, level, seed) for job in jobs for level in args.levels for seed in args.seeds):
        job, level, seed = experiment
        work.put((job, level, seed, slots[index % len(slots)]))
    threads = [threading.Thread(target=worker, args=(work, args, failures), daemon=True) for _ in slots]
    for thread in threads:
        thread.start()
    for _ in threads:
        work.put(None)
    work.join()
    if failures:
        raise RuntimeError(f"{len(failures)} runs failed: {failures[:3]}")


if __name__ == "__main__":
    main()
