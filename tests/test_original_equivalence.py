"""Equivalence of the emulator_bench wrapper with the original Theia code (CPU, fp32).

Run A drives `scripts/train.py` (`train_test_model`, `get_accuracy`) and `drfp.DrfpEncoder` unchanged. Run B is the wrapper
(`cache_features`, then `train.main`). Both use 192 train, 64 val and 64 test real rows of one enzyme retrieval split
(tests/fixtures) and seed 666. The tests compare fingerprints, initial weights, per-epoch train and validation losses, the
early-stopping epoch, final weights and test predictions, and the EC level accuracy code against `src/utils/ec_level_accuracy.py`.
The tests run on one thread: with several threads MKL float sums can differ in the last bit between two runs of equal math.
`scripts/train.py` imports pycm, shap and plotext for reports only, so the test stubs them.
"""

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from drfp import DrfpEncoder
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
SEED = 666
torch.set_num_threads(1)  # MKL sums in a thread-count dependent order, so bitwise equality needs one thread
GROUP = "random_splits_grouped_rxn_key"
FULL_EC = r"\d+\.\d+\.\d+\.\d+"
for path in (ROOT, ROOT.parents[1], ROOT / "src"):
    sys.path.insert(0, str(path))

from emulator_bench import cache_features, train  # noqa: E402
from emulator_bench.common import discover_split_jobs, load_json  # noqa: E402
from emulator_bench.evaluate import ec_level_accuracy  # noqa: E402
from src.utils.ec_level_accuracy import ec_level_accuracy_from_predictions  # noqa: E402


class _ConfusionMatrix:
    """Stand-in for pycm.ConfusionMatrix, which the original only prints."""

    def __init__(self, actual_vector, predict_vector):
        self.overall_stat = {"Overall ACC": 0.0, "Overall MCC": 0.0, "Overall CEN": 0.0}


def load_original_train():
    stubs = {"pycm": types.SimpleNamespace(ConfusionMatrix=_ConfusionMatrix), "shap": types.ModuleType("shap"),
             "plotext": types.SimpleNamespace(clear_figure=lambda: None, clear_data=lambda: None, plot=lambda *a, **k: None, show=lambda: None)}
    with pytest.MonkeyPatch.context() as patch:
        for name, stub in stubs.items():
            patch.setitem(sys.modules, name, stub)
        spec = importlib.util.spec_from_file_location("og_theia_train", ROOT / "scripts" / "train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


ORIGINAL = load_original_train()


def fixture_frames():
    return {name: pd.read_csv(FIXTURES / f"{name}.csv") for name in ("train", "val", "test")}


def original_dataset(frames):
    """Original data path: DrfpEncoder fingerprints, one LabelEncoder over all three label sets, `ReactionDataset`."""
    fps = {name: DrfpEncoder.encode(frame["rxn_smiles"].tolist(), show_progress_bar=False, root_central_atom=False, radius=2,
                                    include_hydrogens=True, n_folded_length=10240) for name, frame in frames.items()}
    encoder = LabelEncoder().fit(pd.concat([frame["ec_number"] for frame in frames.values()]).astype(str))
    datasets = {}
    for name, frame in frames.items():
        df = pd.DataFrame({"rxn_smiles": frame["rxn_smiles"], "label": encoder.transform(frame["ec_number"].astype(str)), "fps": fps[name]})
        datasets[name] = ORIGINAL.ReactionDataset(df)
    return datasets, encoder, fps


def run_original(frames, epochs, patience):
    datasets, encoder, fps = original_dataset(frames)
    torch.manual_seed(SEED)
    initial = ORIGINAL.MLPClassifier(10240, 1664, len(encoder.classes_)).state_dict()
    torch.manual_seed(SEED)
    model, _cm, metrics = ORIGINAL.train_test_model(
        datasets["train"], datasets["val"], datasets["test"], encoder, torch.device("cpu"),
        input_dim=10240, hidden_dim=1664, output_dim=len(encoder.classes_), epochs=epochs, patience=patience,
    )
    with torch.no_grad():
        logits = torch.cat([model(batch) for batch, _ in DataLoader(datasets["test"], batch_size=64)])
    return {"model": model, "metrics": metrics, "initial": initial, "logits": logits, "encoder": encoder, "fps": fps}


def run_wrapper(tmp_path, monkeypatch, frames, epochs, patience):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / GROUP).mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_parquet(dataset_dir / GROUP / f"{name}.parquet", index=False)
    hparams = {**load_json(ROOT / "emulator_bench" / "default_hparams_theia_original.json"), "patience": patience}
    (tmp_path / "hparams.json").write_text(json.dumps(hparams))
    out_root = tmp_path / "out"
    jobs = discover_split_jobs(dataset_dir, [GROUP])
    feature_dir = cache_features.cache_features(out_root, jobs, workers=2)
    argv = ["train", "--dataset_dir", str(dataset_dir), "--out_root", str(out_root), "--hparams_json", str(tmp_path / "hparams.json"),
            "--split_group", GROUP, "--feature_dir", str(feature_dir), "--seed", str(SEED), "--epochs", str(epochs), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()
    return out_root / "enzyme_retrieval" / GROUP / "all" / f"seed_{SEED}", feature_dir


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    frames = fixture_frames()
    torch.use_deterministic_algorithms(True)
    with pytest.MonkeyPatch.context() as patch:
        wrapper, feature_dir = run_wrapper(tmp_path_factory.mktemp("short"), patch, frames, epochs=3, patience=5)
    return {"original": run_original(frames, epochs=3, patience=5), "wrapper": wrapper, "feature_dir": feature_dir, "frames": frames}


def test_fingerprints_match_original(runs):
    packed, index = cache_features.load_features(runs["feature_dir"])
    for name, frame in runs["frames"].items():
        for rxn, original in zip(frame["rxn_smiles"], runs["original"]["fps"][name]):
            assert np.array_equal(np.unpackbits(packed[index[rxn]]), original)


def test_initial_weights_match_original(runs):
    torch.manual_seed(SEED)
    classes = len(runs["original"]["encoder"].classes_)
    initial = train.MLPClassifier(10240, 1664, classes).state_dict()
    assert all(torch.equal(value, runs["original"]["initial"][key]) for key, value in initial.items())


def test_training_matches_original(runs):
    history = pd.read_csv(runs["wrapper"] / "logfile.csv")
    metrics = runs["original"]["metrics"]
    assert len(history) == len(metrics["train_loss"]) == 3
    assert np.allclose(history["train_loss"], metrics["train_loss"], rtol=1e-6, atol=0)
    assert np.allclose(history["valid_loss"], metrics["valid_loss"], rtol=1e-6, atol=0)
    weights = torch.load(runs["wrapper"] / "lastmodel.pth")
    assert all(torch.equal(weights[key], value) for key, value in runs["original"]["model"].state_dict().items())


def test_test_predictions_match_original(runs):
    predictions = pd.read_parquet(runs["wrapper"] / "test_predictions.parquet")
    logits = runs["original"]["logits"]
    classes = runs["original"]["encoder"].classes_
    top = logits.topk(50, dim=1)
    assert [list(ecs) for ecs in predictions["top_ec"]] == classes[top.indices.numpy()].tolist()
    assert np.array_equal(np.stack(predictions["top_score"]), top.values.numpy())


def test_early_stopping_matches_original(tmp_path, monkeypatch):
    """Patience 2 over up to 30 epochs stops the original and the wrapper at the same epoch, with the same last-epoch model."""
    frames = fixture_frames()
    original = run_original(frames, epochs=30, patience=2)
    wrapper, _ = run_wrapper(tmp_path, monkeypatch, frames, epochs=30, patience=2)
    epochs_run = len(pd.read_csv(wrapper / "logfile.csv"))
    assert epochs_run == len(original["metrics"]["valid_loss"]) + 1 < 30
    weights = torch.load(wrapper / "lastmodel.pth")
    assert all(torch.equal(weights[key], value) for key, value in original["model"].state_dict().items())


def test_ec_level_accuracy_matches_src(runs, tmp_path):
    predictions = pd.read_parquet(runs["wrapper"] / "test_predictions.parquet")
    path = tmp_path / "raw.parquet"
    predictions.to_parquet(path, index=False)
    assert ec_level_accuracy(predictions, drop_preliminary=False) == ec_level_accuracy_from_predictions(path)
    full = predictions[predictions["ec_number"].str.fullmatch(FULL_EC)].copy()
    pairs = [[(ec, score) for ec, score in zip(ecs, scores) if re.fullmatch(FULL_EC, ec)] for ecs, scores in zip(full["top_ec"], full["top_score"])]
    full["top_ec"], full["top_score"] = [[ec for ec, _ in row] for row in pairs], [[score for _, score in row] for row in pairs]
    full.to_parquet(path, index=False)
    assert ec_level_accuracy(predictions, drop_preliminary=True) == ec_level_accuracy_from_predictions(path)
