import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch parallel Optuna workers for Theia.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    procs = []
    for index in range(args.workers):
        env = None
        cmd = [sys.executable, str(SCRIPT), *args.args]
        if args.gpus:
            import os

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpus[index % len(args.gpus)])
        procs.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env))
    failed = 0
    for proc in procs:
        failed += int(proc.wait() != 0)
    if failed:
        raise SystemExit(f"{failed} Optuna workers failed")


if __name__ == "__main__":
    main()
