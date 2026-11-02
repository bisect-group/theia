"""Run Theia on all enzyme retrieval splits and seeds, `--runs_per_gpu` runs at once on each GPU."""

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.cache_features import cache_features  # noqa: E402
from emulator_bench.common import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    REPO_ROOT,
    discover_split_jobs,
)

TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Theia on all enzyme retrieval splits and seeds.")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--hparams_json", default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=["0"], help="Physical GPU ids; each run sees one GPU as cuda:0.")
    parser.add_argument("--runs_per_gpu", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast. The original trains in fp32.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = discover_split_jobs(Path(args.dataset_dir), args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.dataset_dir}")
    feature_dir = cache_features(Path(args.out_root), jobs, min(32, os.cpu_count() or 1))

    work = queue.Queue()
    for job in jobs:
        for seed in args.seeds:
            work.put((job, seed))
    failures = []

    def worker(gpu: str) -> None:
        while True:
            try:
                job, seed = work.get_nowait()
            except queue.Empty:
                return
            cmd = [
                sys.executable, str(TRAIN_SCRIPT), "--dataset_dir", args.dataset_dir, "--out_root", args.out_root,
                "--hparams_json", args.hparams_json, "--split_group", job["split_group"], "--split_name", job["split_name"],
                "--feature_dir", str(feature_dir), "--seed", str(seed), "--device", "cuda:0",
            ] + (["--amp"] if args.amp else []) + (["--overwrite"] if args.overwrite else [])
            result = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, cwd=str(REPO_ROOT))
            if result.returncode:
                failures.append(f"{job['split_group']}/{job['split_name']} seed {seed}")

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus for _ in range(args.runs_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        sys.exit("FAILED: " + "; ".join(failures))


if __name__ == "__main__":
    main()
