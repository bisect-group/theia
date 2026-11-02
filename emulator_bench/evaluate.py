"""EC level accuracy on Theia test predictions (same definition as src/utils/ec_level_accuracy.py in EMULaToR)."""

import re

import numpy as np
import pandas as pd

EC_LEVELS = (1, 2, 3, 4)
KS = (1, 3, 5, 10, 20, 30, 40, 50)
FULL_EC = r"\d+\.\d+\.\d+\.\d+"
FULL_EC_RE = re.compile(FULL_EC)
MISS_EC = "0.0.0.0"


def matched_levels(true_ec: str, pred_ec: str) -> int:
    matched = 0
    for pred_part, true_part in zip(pred_ec.split("."), true_ec.split(".")):
        if pred_part != true_part:
            break
        matched += 1
    return matched


def ec_level_accuracy(frame: pd.DataFrame, drop_preliminary: bool) -> dict:
    """Per test reaction and k: share of its true ECs that the top-k ECs match to level n (best prefix match), mean x 100.

    `frame` has one row per test row with `query`, `ec_number`, `top_ec` (best first). With `drop_preliminary`,
    preliminary ECs (such as 3.4.23.B9) leave the truth and the ranked lists, and reactions without a full EC are not scored.
    """
    if drop_preliminary:
        frame = frame[frame["ec_number"].str.fullmatch(FULL_EC)]
    true_ecs, ranked = [], []
    for _, rows in frame.groupby("query", sort=False):
        top = list(rows["top_ec"].iloc[0])
        ranked.append([ec for ec in top if FULL_EC_RE.fullmatch(ec)] if drop_preliminary else top)
        true_ecs.append(sorted(set(rows["ec_number"])))
    result = {}
    for k in KS:
        scores = np.zeros((len(true_ecs), len(EC_LEVELS)))
        for i, (trues, ranking) in enumerate(zip(true_ecs, ranked)):
            top = ranking[:k] or [MISS_EC]
            best = np.array([max(matched_levels(t, p) for p in top) for t in trues])
            scores[i] = [(best >= n).mean() for n in EC_LEVELS]
        means = scores.mean(axis=0) * 100.0
        result[f"k={k}"] = {f"level_{n}_accuracy": float(m) for n, m in zip(EC_LEVELS, means)}
    return {**result, "n_queries": len(true_ecs), "n_rows": len(frame)}
