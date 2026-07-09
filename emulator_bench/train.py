import argparse
import json
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.progress import Progress
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import (
    DEFAULT_DEVICE,
    amp_dtype_for_device,
    configure_torch,
    console,
    load_json,
    read_table,
    resolve_device,
    save_json,
    set_seed,
)
from emulator_bench.evaluate import evaluate_run
from emulator_bench.modeling import CachedReactionDataset, build_model, predict_logits


def autocast_context(device: torch.device, dtype):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_hparams(path: Path) -> dict:
    hparams = load_json(path)
    return {
        "input_dim": int(hparams.get("input_dim", 10240)),
        "hidden_dim": int(hparams.get("hidden_dim", 1664)),
        "batch_size": int(hparams.get("batch_size", 64)),
        "epochs": int(hparams.get("epochs", 50)),
        "patience": int(hparams.get("patience", 5)),
        "min_delta": float(hparams.get("min_delta", 0.001)),
        "lr": float(hparams.get("lr", 0.001)),
        "scheduler_gamma": float(hparams.get("scheduler_gamma", 0.9)),
    }


def add_encoded_labels(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([train["label"], val["label"], test["label"]]).astype(str))
    for frame in (train, val, test):
        frame["encoded_label"] = label_encoder.transform(frame["label"].astype(str))
    return label_encoder


def train_one(args) -> dict:
    configure_torch()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics.json").exists() and not args.overwrite:
        console.print(f"Run already complete: {out_dir}")
        return load_json(out_dir / "metrics.json")

    train_df = read_table(Path(args.train_path))
    val_df = read_table(Path(args.val_path))
    test_df = read_table(Path(args.test_path))
    label_encoder = add_encoded_labels(train_df, val_df, test_df)
    hparams = load_hparams(Path(args.hparams_json))
    save_json(out_dir / "hparams.json", hparams)
    with open(out_dir / "label_encoder.pkl", "wb") as handle:
        pickle.dump(label_encoder, handle)

    device = resolve_device(args.device)
    amp_dtype, precision_mode = amp_dtype_for_device(device)
    model = build_model(hparams["input_dim"], hparams["hidden_dim"], len(label_encoder.classes_)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hparams["scheduler_gamma"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype is torch.float16))
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "precision_mode": precision_mode}
    checkpoint = out_dir / "checkpoint_last.pt"
    if checkpoint.exists() and not args.overwrite:
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_val_loss = float(payload["best_val_loss"])
        history = payload.get("history", history)
        console.print(f"Resuming {out_dir} from epoch {start_epoch}")

    train_loader = DataLoader(CachedReactionDataset(train_df), batch_size=hparams["batch_size"], shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(CachedReactionDataset(val_df), batch_size=hparams["batch_size"], shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    epochs_without_improvement = 0
    with Progress() as progress:
        epoch_task = progress.add_task(f"Training {args.level}", total=hparams["epochs"] - start_epoch)
        for epoch in range(start_epoch, hparams["epochs"]):
            model.train()
            batch_losses = []
            batch_task = progress.add_task(f"Epoch {epoch}", total=len(train_loader))
            for data, target, _indices in train_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, amp_dtype):
                    logits = model(data)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_losses.append(float(loss.detach().cpu()))
                progress.advance(batch_task)
            progress.remove_task(batch_task)
            scheduler.step()
            train_true, train_pred, _idx, train_eval_loss = predict_logits(model, train_loader, device)
            val_true, val_pred, _idx, val_loss = predict_logits(model, val_loader, device)
            train_acc = float(np.mean(np.asarray(train_true) == np.asarray(train_pred))) if train_true else 0.0
            val_acc = float(np.mean(np.asarray(val_true) == np.asarray(val_pred))) if val_true else 0.0
            history["train_loss"].append(float(np.mean(batch_losses)) if batch_losses else train_eval_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            improved = val_loss < best_val_loss - hparams["min_delta"]
            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), out_dir / "best_model.pt")
            else:
                epochs_without_improvement += 1
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                checkpoint,
            )
            save_json(out_dir / "history.json", history)
            progress.advance(epoch_task)
            if len(history["val_loss"]) >= hparams["patience"] and epochs_without_improvement >= hparams["patience"]:
                break

    if not (out_dir / "best_model.pt").exists():
        torch.save(model.state_dict(), out_dir / "best_model.pt")
    return evaluate_run(out_dir, Path(args.test_path), args.device, hparams["batch_size"], args.level)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Theia EMULaToR classifier.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--level", default="ec1234")
    parser.add_argument("--hparams_json", default=str(Path(__file__).resolve().parent / "default_hparams_theia_original.json"))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metrics = train_one(args)
    console.print(metrics)


if __name__ == "__main__":
    main()
