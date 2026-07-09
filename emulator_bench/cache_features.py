import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_LEVELS,
    REQUIRED_COLUMNS,
    SplitJob,
    atomic_save_npz,
    cached_split_path,
    console,
    copy_dataset_splits,
    discover_split_jobs,
    feature_cache_path,
    normalize_rxn_smiles,
    read_table,
    require_columns,
    save_json,
    write_table,
)
from emulator_bench.ec import ec_label, is_unlabeled_ec


def _encode_batch(batch: List[str]):
    fps = DrfpEncoder.encode(
        batch,
        show_progress_bar=False,
        root_central_atom=False,
        radius=2,
        include_hydrogens=True,
        n_folded_length=10240,
    )
    return [(rxn, np.asarray(fp, dtype=np.uint8)) for rxn, fp in zip(batch, fps)]


def _batches(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def encode_reactions(
    reactions: List[str],
    embeddings_dir: Path,
    overwrite: bool = False,
    batch_size: int = 1024,
    num_workers: int = 1,
) -> Dict[str, int]:
    unique = sorted({normalize_rxn_smiles(value) for value in reactions if str(value).strip()})
    pending = [rxn for rxn in unique if overwrite or not feature_cache_path(embeddings_dir, rxn).exists()]
    written = 0
    batches = _batches(pending, max(1, int(batch_size)))
    with Progress() as progress:
        task = progress.add_task("Caching DRFP fingerprints", total=len(pending))
        if num_workers <= 1 or len(batches) <= 1:
            encoded_iter = (_encode_batch(batch) for batch in batches)
            for encoded in encoded_iter:
                for rxn, fp in encoded:
                    atomic_save_npz(feature_cache_path(embeddings_dir, rxn), fp=fp)
                    written += 1
                    progress.advance(task)
        else:
            with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
                futures = [executor.submit(_encode_batch, batch) for batch in batches]
                for future in as_completed(futures):
                    encoded = future.result()
                    for rxn, fp in encoded:
                        atomic_save_npz(feature_cache_path(embeddings_dir, rxn), fp=fp)
                        written += 1
                        progress.advance(task)
    return {"unique_reactions": len(unique), "written": written, "already_cached": len(unique) - len(pending), "num_workers": int(num_workers)}


def ensure_reactions_cached(reactions: List[str], embeddings_dir: Path) -> None:
    missing = []
    for rxn in reactions:
        path = feature_cache_path(embeddings_dir, rxn)
        if not path.exists():
            missing.append(str(path))
            if len(missing) >= 5:
                break
    if missing:
        raise FileNotFoundError(f"Missing cached DRFP files. First missing paths: {missing}")


def load_and_filter_split(path: Path, level: str, embeddings_dir: Path, max_rows: int = 0) -> pd.DataFrame:
    df = read_table(path)
    require_columns(df, REQUIRED_COLUMNS, path)
    if max_rows:
        df = df.head(max_rows).copy()
    rows = []
    for row_index, row in df.reset_index(drop=True).iterrows():
        ec = str(row["ec_number"])
        if is_unlabeled_ec(ec):
            continue
        try:
            label = ec_label(ec, level)
        except ValueError:
            continue
        rxn = normalize_rxn_smiles(row["rxn_smiles"])
        rows.append(
            {
                "row_id": f"{path}:{row_index}",
                "rxn_smiles": rxn,
                "ec_number": ec,
                "label": label,
                "feature_path": str(feature_cache_path(embeddings_dir, rxn)),
            }
        )
    cached = pd.DataFrame(rows)
    if cached.empty:
        raise ValueError(f"No usable {level} labels found in {path}")
    return cached


def cache_job(
    job: SplitJob,
    levels: List[str],
    embeddings_dir: Path,
    overwrite: bool = False,
    max_rows: int = 0,
    batch_size: int = 1024,
    num_workers: int = 1,
) -> None:
    frames = {name: read_table(path) for name, path in [("train", job.train_path), ("val", job.val_path), ("test", job.test_path)]}
    all_reactions = []
    for path, frame in [(job.train_path, frames["train"]), (job.val_path, frames["val"]), (job.test_path, frames["test"])]:
        require_columns(frame, REQUIRED_COLUMNS, path)
        if max_rows:
            frame = frame.head(max_rows)
        all_reactions.extend(frame["rxn_smiles"].dropna().astype(str).tolist())
    summary = encode_reactions(all_reactions, embeddings_dir, overwrite=overwrite, batch_size=batch_size, num_workers=num_workers)
    ensure_reactions_cached([normalize_rxn_smiles(value) for value in all_reactions], embeddings_dir)
    for level in levels:
        for split, path in [("train", job.train_path), ("val", job.val_path), ("test", job.test_path)]:
            output = cached_split_path(job, level, split)
            if output.exists() and not overwrite:
                continue
            cached = load_and_filter_split(path, level, embeddings_dir, max_rows=max_rows)
            missing = [fp for fp in cached["feature_path"] if not Path(fp).exists()]
            if missing:
                raise FileNotFoundError(f"Missing cached feature file: {missing[0]}")
            write_table(cached, output)
    save_json(job.root_dir / "theia_cache" / "cache_summary.json", {"job": job.as_dict(), **summary, "levels": levels})


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Theia DRFP features for EMULaToR splits.")
    parser.add_argument("--source_dataset_dir", default="")
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--split_groups", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy_splits", action="store_true")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel DRFP encoding worker processes.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser()
    embeddings_dir = Path(args.embeddings_dir).expanduser()
    if args.copy_splits:
        source = Path(args.source_dataset_dir).expanduser() if args.source_dataset_dir else None
        if source is None:
            raise ValueError("--source_dataset_dir is required with --copy_splits")
        copied = copy_dataset_splits(source, base_dir, overwrite=args.overwrite)
        console.print(f"Copied {len(copied)} split/data files")

    jobs = discover_split_jobs(base_dir, split_groups=args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found under {base_dir}")
    for job in jobs:
        console.rule(f"{job.split_group}/{job.split_name}")
        cache_job(
            job,
            args.levels,
            embeddings_dir,
            overwrite=args.overwrite,
            max_rows=args.max_rows,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
