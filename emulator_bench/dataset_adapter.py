import argparse
import sys
from pathlib import Path

from rich.progress import track

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_SOURCE_DATA_DIR,
    REQUIRED_COLUMNS,
    console,
    copy_dataset_splits,
    discover_split_jobs,
    read_table,
    render_jobs_table,
    require_columns,
)


def validate_jobs(base_dir: Path) -> None:
    jobs = discover_split_jobs(base_dir)
    if not jobs:
        raise FileNotFoundError(f"No train/val/test split jobs found under {base_dir}")
    for job in track(jobs, description="Validating copied splits"):
        for path in (job.train_path, job.val_path, job.test_path):
            df = read_table(path)
            require_columns(df, REQUIRED_COLUMNS, path)
            if df.empty:
                raise ValueError(f"Empty split file: {path}")
    render_jobs_table(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy and validate EMULaToR enzyme retrieval splits for Theia.")
    parser.add_argument("--source_dataset_dir", default=str(DEFAULT_SOURCE_DATA_DIR))
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    copied = copy_dataset_splits(Path(args.source_dataset_dir), Path(args.base_dir), overwrite=args.overwrite)
    console.print(f"Copied {len(copied)} files into {args.base_dir}")
    validate_jobs(Path(args.base_dir))


if __name__ == "__main__":
    main()
