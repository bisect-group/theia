"""DRFP reaction fingerprints, computed once per unique reaction SMILES for all discovered splits.

The store is `<out_root>/embeddings/<FEATURE_TAG>/<key-set hash>/{fps.npy,keys.parquet}`.
`fps.npy` holds the 10240-bit fingerprints packed to 1280 bytes per row, in the order of `keys.parquet`.
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_OUT_ROOT,
    DEFAULT_SPLIT_GROUPS,
    atomic_write,
    discover_split_jobs,
    read_split,
)

# Theia scripts/encode_split_data.py: DrfpEncoder.encode(radius=2, include_hydrogens=True, root_central_atom=False, n_folded_length=10240).
FEATURE_TAG = "drfp_v1_r2_n10240_h1_root0"
CHUNK = 500


def _encode(reactions: List[str]) -> np.ndarray:
    from drfp import DrfpEncoder

    fps = DrfpEncoder.encode(reactions, show_progress_bar=False, root_central_atom=False, radius=2, include_hydrogens=True, n_folded_length=10240)
    return np.packbits(np.asarray(fps, dtype=np.uint8), axis=1)


def split_reactions(jobs) -> List[str]:
    unique = set()
    for job in jobs:
        for stem in ("train", "val", "test"):
            unique.update(read_split(job[f"{stem}_path"])["rxn_smiles"].astype(str).str.strip())
    return sorted(unique)


def feature_dir(out_root: Path, reactions: List[str]) -> Path:
    fingerprint = hashlib.sha256("\n".join(reactions).encode()).hexdigest()[:12]
    return Path(out_root) / "embeddings" / FEATURE_TAG / fingerprint


def cache_features(out_root: Path, jobs, workers: int) -> Path:
    reactions = split_reactions(jobs)
    directory = feature_dir(out_root, reactions)
    if (directory / "keys.parquet").exists():
        return directory
    chunks = [reactions[start : start + CHUNK] for start in range(0, len(reactions), CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        packed = np.concatenate(list(pool.map(_encode, chunks)))
    atomic_write(directory / "fps.npy", lambda tmp: np.save(open(tmp, "wb"), packed))
    atomic_write(directory / "keys.parquet", lambda tmp: pd.DataFrame({"rxn_smiles": reactions}).to_parquet(tmp, index=False))
    print(f"Cached {len(reactions)} unique reactions in {directory}", flush=True)
    return directory


def load_features(directory: Path):
    """Returns the packed fingerprints and a dict from reaction SMILES to row."""
    keys = pd.read_parquet(Path(directory) / "keys.parquet")["rxn_smiles"]
    return np.load(Path(directory) / "fps.npy"), {rxn: row for row, rxn in enumerate(keys)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Theia DRFP fingerprints for the EMULaToR enzyme retrieval splits.")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    jobs = discover_split_jobs(Path(args.dataset_dir), args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.dataset_dir}")
    print(cache_features(Path(args.out_root), jobs, args.workers))


if __name__ == "__main__":
    main()
