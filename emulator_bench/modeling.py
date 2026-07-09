import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from emulator_bench.common import REPO_ROOT, SRC_ROOT

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from theia.ml import MLPClassifier  # noqa: E402


class CachedReactionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, label_col: str = "encoded_label"):
        self.frame = frame.reset_index(drop=True)
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with np.load(Path(row["feature_path"]), allow_pickle=False) as data:
            fp = data["fp"].astype(np.float32, copy=False)
        return torch.from_numpy(fp), torch.tensor(int(row[self.label_col]), dtype=torch.long), index


def build_model(input_dim: int, hidden_dim: int, output_dim: int) -> MLPClassifier:
    return MLPClassifier(input_dim, hidden_dim, output_dim)


def predict_logits(model, loader, device: torch.device) -> tuple:
    model.eval()
    labels: List[int] = []
    preds: List[int] = []
    indices: List[int] = []
    losses: List[float] = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target, batch_indices in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(data)
            loss = criterion(logits, target)
            pred = logits.argmax(dim=1)
            labels.extend(target.cpu().tolist())
            preds.extend(pred.cpu().tolist())
            indices.extend(batch_indices.cpu().tolist())
            losses.append(float(loss.item()))
    return labels, preds, indices, float(np.mean(losses)) if losses else 0.0
