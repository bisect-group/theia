import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import DEFAULT_DEVICE, console, read_table, save_json
from emulator_bench.ec import build_level_vocab, hierarchy_accuracies
from emulator_bench.modeling import CachedReactionDataset, build_model, predict_logits


def evaluate_run(run_dir: Path, split_path: Path, device_name: str = DEFAULT_DEVICE, batch_size: int = 256, level: str = "ec1234") -> dict:
    run_dir = Path(run_dir)
    split = read_table(split_path)
    with open(run_dir / "label_encoder.pkl", "rb") as handle:
        label_encoder = pickle.load(handle)
    split["encoded_label"] = label_encoder.transform(split["label"].astype(str))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = build_model(10240, int(json.loads((run_dir / "hparams.json").read_text())["hidden_dim"]), len(label_encoder.classes_))
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.to(device)
    loader = DataLoader(CachedReactionDataset(split), batch_size=batch_size, shuffle=False, num_workers=0)
    y_true_idx, y_pred_idx, indices, loss = predict_logits(model, loader, device)
    y_true = label_encoder.inverse_transform(y_true_idx)
    y_pred = label_encoder.inverse_transform(y_pred_idx)
    ordered = split.iloc[indices].copy()
    vocab_by_level = {name: build_level_vocab(ordered["ec_number"].astype(str), name) for name in ("ec1", "ec12", "ec123", "ec1234")}
    metrics = {
        "loss": loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else 0.0,
        "n_samples": int(len(y_true)),
        "level": level,
    }
    metrics.update(hierarchy_accuracies(ordered["ec_number"].astype(str).tolist(), list(y_pred), vocab_by_level=vocab_by_level))
    predictions = ordered[["row_id", "rxn_smiles", "ec_number", "label"]].copy()
    predictions["prediction"] = y_pred
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=label_encoder.classes_), index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(run_dir / "confusion_matrix.csv")
    save_json(run_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Theia EMULaToR bench run.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--level", default="ec1234")
    args = parser.parse_args()
    metrics = evaluate_run(Path(args.run_dir), Path(args.split_path), args.device, args.batch_size, args.level)
    console.print(metrics)


if __name__ == "__main__":
    main()
