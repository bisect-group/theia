# Theia emulator_bench

This wrapper trains Theia (reaction to EC number) on the default EMULaToR enzyme retrieval splits with the original settings. It runs seeds 666, 777 and 888 on every split. It does not tune anything. The original Theia code is not changed.

## Data

Default dataset: `data/processed/datasets/enzyme_retrieval_dataset` (reaction-only splits, not `with_protein_information`).
The wrapper reads these splits and never writes into the data folder:

- `random_splits_grouped_rxn_key`
- `uniprot_time_splits`
- `reaction_drfp_tanimoto_splits/threshold_*`

Each split has `train.parquet`, `val.parquet` and `test.parquet` with `rxn_smiles` and `ec_number`.

## Method and settings

Theia turns each reaction into a 10240-bit DRFP fingerprint (radius 2, hydrogens, no central-atom root).
A two-layer MLP (`theia.ml.MLPClassifier`: 10240 to 1664, tanh, 1664 to the number of ECs) predicts the EC.
The label is the full four-part EC number. The settings are in `default_hparams_theia_original.json`, with the source of each value:
Adam lr 0.001, ExponentialLR gamma 0.9, batch size 64, at most 50 epochs, early stopping on the mean of the last 5 validation losses.
As in the original, batches follow the stored row order (no shuffle), the label encoder sees the labels of all three splits,
and the test pass uses the model of the last epoch. Training is fp32.

## Cache

`cache_features.py` computes the fingerprints once per unique reaction SMILES of all discovered splits.
The store is `outputs/baselines/Theia/embeddings/drfp_v1_r2_n10240_h1_root0/<key-set hash>/`. It holds `fps.npy` (bit-packed) and `keys.parquet`.
A different set of reactions gives a different hash, so a changed dataset builds a new store.

## Outputs

`outputs/baselines/Theia/enzyme_retrieval/<split_group>/<split_name>/seed_<seed>/` holds:

- `test_predictions.parquet`: one row per test row with `query` (the reaction), `ec_number`, `top_ec` and `top_score` (top 50 ECs, best first, logits).
- `ec_level_accuracy_no_b.json`: EC level accuracy at k = 1, 3, 5, 10, 20, 30, 40, 50 and levels 1 to 4. Preliminary ECs (such as `3.4.23.B9`) leave the truth and the ranked lists. Reactions without a full EC are not scored. This is the score to report.
- `ec_level_accuracy.json`: the same without the preliminary-EC filter.
- `metrics.json`, `logfile.csv` (loss per epoch), `lastmodel.pth`.

`aggregate_results.py` writes `summary_runs.csv` and `summary_by_split.csv` (mean and std over seeds, k = 1) in `outputs/baselines/Theia/enzyme_retrieval/`.

## Commands

See `commands.txt`. Run them from the submodule root (`baselines/theia`).
`verify_equivalence.sh` checks the wrapper against the original code on a small real fixture (CPU).
