"""Train and evaluate Theia (reaction -> EC) on one EMULaToR enzyme retrieval split with the original settings.

The loop follows scripts/train.py of Theia: `MLPClassifier`, Adam, ExponentialLR, batches of 64 in stored row order
(the original DataLoader does not shuffle), early stopping on the mean of the last `patience` validation losses,
and the test pass on the model of the last epoch. The label is the full EC number.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.cache_features import load_features  # noqa: E402
from emulator_bench.common import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_OUT_ROOT,
    DEFAULT_SPLIT_GROUPS,
    SRC_ROOT,
    atomic_write,
    discover_split_jobs,
    load_json,
    read_split,
    run_dir,
    save_json,
    set_seed,
)
from emulator_bench.evaluate import KS, ec_level_accuracy  # noqa: E402

sys.path.insert(0, str(SRC_ROOT))
from theia.ml import MLPClassifier  # noqa: E402

TOP_K = max(KS)


def to_device(frame: pd.DataFrame, packed: np.ndarray, index: dict, device) -> torch.Tensor:
    """Fingerprints of the rows as bit-unpacked uint8 on the device. Batches are converted to float32."""
    rows = frame["rxn_smiles"].astype(str).str.strip().map(index).to_numpy()
    return torch.from_numpy(np.unpackbits(packed[rows], axis=1)).to(device)


def batches(count: int, size: int):
    return [slice(start, min(start + size, count)) for start in range(0, count, size)]


def evaluate_loss(model, x, y, batch_size: int, criterion, amp_dtype, device) -> float:
    """Mean over batches of the batch mean loss, as in the original validation pass."""
    model.eval()
    total = torch.zeros((), device=device, dtype=torch.float64)
    with torch.no_grad():
        for rows in batches(len(y), batch_size):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x[rows].float())
            total += criterion(logits.float(), y[rows]).double()
    return float(total) / len(batches(len(y), batch_size))


def test_predictions(model, x, frame: pd.DataFrame, classes: np.ndarray, batch_size: int, amp_dtype, device) -> pd.DataFrame:
    model.eval()
    scores, indices = [], []
    with torch.no_grad():
        for rows in batches(len(x), batch_size):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x[rows].float())
            top = logits.float().topk(min(TOP_K, logits.shape[1]), dim=1)
            scores.append(top.values.cpu())
            indices.append(top.indices.cpu())
    scores, indices = torch.cat(scores).numpy(), torch.cat(indices).numpy()
    return pd.DataFrame(
        {
            "query": frame["rxn_smiles"].to_numpy(),
            "ec_number": frame["ec_number"].to_numpy(),
            "top_ec": classes[indices].tolist(),
            "top_score": scores.tolist(),
        }
    )


def train_one(args, job: dict, feature_dir: Path) -> dict:
    out = run_dir(args.out_root, job["split_group"], job["split_name"], args.seed)
    if (out / "metrics.json").exists() and not args.overwrite:
        return load_json(out / "metrics.json")
    out.mkdir(parents=True, exist_ok=True)
    hparams = load_json(args.hparams_json)
    if args.epochs is not None:
        hparams["epochs"] = args.epochs
    set_seed(args.seed)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if args.amp and device.type == "cuda" else None

    frames = {name: read_split(job[f"{name}_path"]) for name in ("train", "val", "test")}
    if args.sample_rows:
        frames = {name: frame.head(args.sample_rows) for name, frame in frames.items()}
    # The original fits the label encoder on the labels of all three splits.
    encoder = LabelEncoder().fit(pd.concat([frame["ec_number"] for frame in frames.values()]).astype(str))
    packed, index = load_features(feature_dir)
    x = {name: to_device(frame, packed, index, device) for name, frame in frames.items()}
    y = {name: torch.from_numpy(encoder.transform(frame["ec_number"].astype(str))).long().to(device) for name, frame in frames.items()}

    model = MLPClassifier(hparams["input_dim"], hparams["hidden_dim"], len(encoder.classes_)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hparams["scheduler_gamma"])
    batch_size = hparams["batch_size"]
    n_train_batches = len(batches(len(y["train"]), batch_size))

    started = time.time()
    history = []
    valid_losses = []
    for epoch in range(hparams["epochs"]):
        model.train()
        train_loss = torch.zeros((), device=device, dtype=torch.float64)
        for rows in batches(len(y["train"]), batch_size):
            optimizer.zero_grad()
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x["train"][rows].float())
            loss = criterion(logits.float(), y["train"][rows])
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().double()
        valid_loss = evaluate_loss(model, x["val"], y["val"], batch_size, criterion, amp_dtype, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": float(train_loss) / n_train_batches, "valid_loss": valid_loss, "seconds": time.time() - started})
        print(history[-1], flush=True)
        if len(valid_losses) >= hparams["patience"] and valid_loss - np.mean(valid_losses[-hparams["patience"] :]) > -hparams["min_delta"]:
            break
        valid_losses.append(valid_loss)

    torch.save(model.state_dict(), out / "lastmodel.pth")
    predictions = test_predictions(model, x["test"], frames["test"], encoder.classes_, batch_size, amp_dtype, device)
    atomic_write(out / "test_predictions.parquet", lambda tmp: predictions.to_parquet(tmp, index=False))
    accuracy = ec_level_accuracy(predictions, drop_preliminary=False)
    accuracy_no_preliminary = ec_level_accuracy(predictions, drop_preliminary=True)
    save_json(out / "ec_level_accuracy.json", accuracy)
    save_json(out / "ec_level_accuracy_no_b.json", accuracy_no_preliminary)
    pd.DataFrame(history).to_csv(out / "logfile.csv", index=False)
    metrics = {
        "split": f"{job['split_group']}/{job['split_name']}",
        "seed": args.seed,
        "epochs_run": len(history),
        "n_classes": len(encoder.classes_),
        "rows": {name: len(frame) for name, frame in frames.items()},
        "precision": "bf16" if amp_dtype else "fp32",
        "elapsed_seconds": time.time() - started,
        "ec_level_accuracy_no_b": accuracy_no_preliminary,
        "ec_level_accuracy": accuracy,
        "hparams": hparams,
    }
    save_json(out / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Theia on one enzyme retrieval split and seed.")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--hparams_json", default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--split_group", required=True, choices=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--split_name", default="all", help="`all` for flat groups, else a `threshold_*` folder.")
    parser.add_argument("--feature_dir", required=True, help="Folder made by cache_features.py.")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA. The original trains in fp32.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Test-only override of the epoch count.")
    parser.add_argument("--sample_rows", type=int, default=0, help="Test-only: use the first rows of each split.")
    args = parser.parse_args()
    job = next(j for j in discover_split_jobs(Path(args.dataset_dir), [args.split_group]) if j["split_name"] == args.split_name)
    print(train_one(args, job, Path(args.feature_dir))["ec_level_accuracy_no_b"]["k=1"])


if __name__ == "__main__":
    main()
