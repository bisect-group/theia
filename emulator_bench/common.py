import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EMULATOR_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = EMULATOR_ROOT / "data" / "processed" / "datasets" / "enzyme_retrieval_dataset"
DEFAULT_OUT_ROOT = EMULATOR_ROOT / "outputs" / "baselines" / "Theia"
DEFAULT_HPARAMS_PATH = REPO_ROOT / "emulator_bench" / "default_hparams_theia_original.json"
TASK = "enzyme_retrieval"
DEFAULT_SPLIT_GROUPS = ["random_splits_grouped_rxn_key", "uniprot_time_splits", "reaction_drfp_tanimoto_splits"]
DEFAULT_SEEDS = [666, 777, 888]
REQUIRED_COLUMNS = ("rxn_smiles", "ec_number")


def load_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text())


def atomic_write(path: Path, write) -> None:
    """Call `write(tmp_path)`, then move the temp file (in the target folder) onto `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    write(tmp)
    tmp.replace(path)


def save_json(path: Path, payload: Dict) -> None:
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")
    return frame


def split_file(directory: Path, stem: str):
    path = Path(directory) / f"{stem}.parquet"
    return path if path.exists() else None


def discover_split_jobs(dataset_dir: Path, split_groups: List[str]) -> List[Dict[str, Path]]:
    """Flat groups give split_name `all`; groups with `threshold_*` folders give one job per folder."""
    jobs = []
    for group in split_groups:
        group_dir = Path(dataset_dir) / group
        candidates = [("all", group_dir)] + [(d.name, d) for d in sorted(group_dir.glob("threshold_*")) if d.is_dir()]
        for name, directory in candidates:
            paths = {stem: split_file(directory, stem) for stem in ("train", "val", "test")}
            if all(paths.values()):
                jobs.append({"split_group": group, "split_name": name, **{f"{stem}_path": path for stem, path in paths.items()}})
    return jobs


def run_dir(out_root: Path, split_group: str, split_name: str, seed: int) -> Path:
    return Path(out_root) / TASK / split_group / split_name / f"seed_{seed}"
