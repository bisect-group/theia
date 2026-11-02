# Theia emulator_bench report

## Method
Theia classifies a reaction into a full EC number. The input is a DRFP fingerprint (10240 bits). The model is a two-layer tanh MLP (hidden 1664).
The wrapper retrains it from scratch (no released weights) on EMULaToR enzyme retrieval splits, with 3 seeds.
The score for each test reaction is the EC level accuracy of the top-50 ranked ECs. Preliminary B-ECs are dropped from the truth and the ranked lists for `ec_level_accuracy_no_b.json`.

## Settings and sources
| Setting | Value | Source |
|---|---|---|
| fingerprint | DRFP radius 2, hydrogens, no root atom, 10240 bits | scripts/encode_split_data.py |
| hidden size | 1664 | scripts/train.py:224 |
| batch size | 64, no shuffle | scripts/train.py:245-246,269 |
| optimizer | Adam lr 1e-3, no weight decay | scripts/train.py:231 |
| scheduler | ExponentialLR gamma 0.9 per epoch | scripts/train.py:232 |
| epochs | up to 50 | scripts/train.py main |
| early stop | valid loss minus mean of last 5 > -0.001 | scripts/train.py:255-258 |
| model for test | last epoch | scripts/train.py |
| precision | fp32 | no AMP in the original |
The values and line numbers are also in `default_hparams_theia_original.json`.

## Wrapper
- `common.py`, `cache_features.py`: DRFP cache, once per unique reaction (74688 reactions, bit-packed).
- `train.py`: the original loop on GPU-resident fingerprints. Writes `lastmodel.pth`, `test_predictions.parquet` (top-50), `ec_level_accuracy.json`, `ec_level_accuracy_no_b.json`, `logfile.csv`, `metrics.json`.
- `run_split_benchmarks.py`: builds the cache, then queues splits x seeds 666 777 888 (9 runs: random grouped by reaction, uniprot time, reaction DRFP Tanimoto).
- `aggregate_results.py`: mean and std over seeds.
- The output layout and `test_predictions.parquet` match the MLP baseline.

## Equivalence with the original (tests/test_original_equivalence.py)
Data: 192/64/64 real rows, seed 666, CPU, fp32.
- Fingerprints are bit-equal to `DrfpEncoder`. Initial weights are equal.
- Per-epoch train and valid losses are within rtol 1e-6. Final weights and top-50 test scores are bitwise equal.
- Early stopping (patience 2, up to 30 epochs) stops at the same epoch with the same weights.
- EC level accuracy equals `src/utils/ec_level_accuracy.py`.
- 7 passed locally (`bash emulator_bench/verify_equivalence.sh`).

## Speed-ups
Exact: fingerprints in GPU memory (bit-packed on disk), no DataLoader, float64 loss sum on the GPU. The cache build is parallel over processes.

## Precision
fp32. bf16 `--amp` gave no gain (5.3 s against 5.4 s per epoch, random split), so the default stays fp32.

## Time on RTX PRO 4500 Blackwell (one run alone, fp32)
- random_splits_grouped_rxn_key: 5.2 s per epoch, 22 epochs, 115 s. EC level accuracy no-B k=1 level 4: 44.9.
- uniprot_time_splits: 5.5 s per epoch, 6 epochs, 34 s. Level 4: 12.8.
- The cache build took a few minutes on the host CPUs.
- 9 runs, worst case 50 epochs: under 40 min in sequence. Expected total under 20 min.

## Runs per GPU and ETA
3 runs per GPU on GPU 0. Total about 15 min after the cache. Start any time before Mon 2 Nov 04:00 IST. The cache is already built on the host.

## Launch
```
cd ~/rerun4/Theia/baselines/theia
$PY emulator_bench/run_split_benchmarks.py --gpus 0 --runs_per_gpu 3
$PY emulator_bench/aggregate_results.py
```

## Open issues
- DECISION NEEDED: one full-EC classifier is trained (the original `ec1234` model). The original also trains separate ec1, ec12 and ec123 classifiers. They are not run. Level accuracy comes from the full-EC ranking.
- Test query count can differ from the MLP baseline because rows without a full-EC label are not in the test files.
- Stale files from the old wrapper remain: `dataset_adapter.py`, `ec.py`, `tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`, `modeling.py`.
