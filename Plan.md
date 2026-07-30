# Theia EMULaToR Adaptation Plan

## 1. Baseline Summary From README

- Baseline repository: `/home/adhil/github/theia`
- Task addressed by the baseline: reaction SMILES to EC number classification with DRFP fingerprints and a PyTorch MLP.
- README and docs inspected: `README.md`, `scripts/encode_split_data.py`, `scripts/train.py`, `train_all.sh`, `src/theia/ml/*`.
- Recommended training command from docs: encode CSV data with `python scripts/encode_split_data.py ...`, then train with `python scripts/train.py <train.csv> <valid.csv> <test.csv> <output>`.
- Recommended evaluation command from docs: evaluation is part of `scripts/train.py`; it writes confusion-matrix and metric artifacts after training.
- Default config or hyperparameter source: paper/README and `scripts/train.py`: DRFP length `10240`, radius `2`, `root_central_atom=False`, `include_hydrogens=True`, hidden size `1664`, Adam `lr=0.001`, exponential scheduler `gamma=0.9`, batch size `64`, max `50` epochs, patience `5`, min validation improvement `0.001`.
- Notes about any doc or code mismatch: Theia's released workflow trains EC1, EC12, and EC123. This bench also adds EC1234 because EMULaToR labels include full four-part EC numbers.

## 2. Expected Input Format

- Raw file format(s): parquet or CSV split files.
- Required columns or fields: `rxn_smiles`, `ec_number`; `uniprot_date` is preserved when present.
- Label or target fields: `ec_number`, mapped to `ec1`, `ec12`, `ec123`, or `ec1234`.
- Identifier fields: no stable ID column exists in the source splits, so the bench derives deterministic row IDs from split path, row number, reaction SMILES, and EC number.
- Any assumptions about train, validation, and test partitioning: split folders contain `train`, `val`, and `test` files as parquet or CSV.

## 3. Featurization and Preprocessing Path

- Native preprocessing entrypoint: `scripts/encode_split_data.py`.
- Native featurization functions, scripts, or modules: `drfp.DrfpEncoder.encode` with Theia's paper settings.
- Required intermediate artifacts: split-specific cached tables plus one shared DRFP cache per unique reaction.
- Where cached features will be stored: `~/github/EMULaToR/data/processed/baselines/theia/embeddings/reactions/drfp_r2_10240_h1_root0/`.
- Cache format for train, validation, and test: split tables under each split directory in `theia_cache/<level>/<split>.parquet`, with `feature_path` pointers to compressed `.npz` fingerprints.

## 4. Training and Evaluation Entrypoints

- Training entrypoint: `emulator_bench/train.py`.
- Evaluation entrypoint: `emulator_bench/evaluate.py`, also run automatically at the end of training.
- Checkpoint location or discovery rule: each run directory stores `checkpoint_last.pt`, `best_model.pt`, `label_encoder.pkl`, `metrics.json`, and `predictions.csv`.
- Default settings that must remain unchanged: architecture and paper/default optimizer settings from Theia are stored in `emulator_bench/default_hparams_theia_original.json`.
- Exact wrapper-to-baseline command mapping: the wrapper consumes EMULaToR split files, loads cached DRFP fingerprints, instantiates `theia.ml.MLPClassifier`, and trains with Theia's default loss, optimizer, scheduler, and early stopping behavior.

## 5. Dataset and Split Mapping

- User dataset path: `~/github/EMULaToR/data/processed/datasets/enzyme_retrieval_dataset`.
- User split definition path or files: `random_splits`, `uniprot_time_splits`, and `reaction_drfp_tanimoto_splits/threshold_*`.
- Mapping from user schema to baseline schema: `rxn_smiles -> rxn_smiles`, `ec_number -> label` after EC-level projection.
- Mapping from user train, validation, and test splits to baseline expectations: each split file maps directly to a cached Theia split table.
- Assumptions or blockers: rows with unlabeled `-.-.-.-` are excluded from training/evaluation for a concrete EC level; partially specified labels are kept for hierarchy evaluation when their known prefix is valid.

## 6. Files To Add Under `emulator_bench/`

- `README.md`: documents inputs, caching, commands, and enhancements.
- `dataset_adapter.py`: copies and validates EMULaToR split trees.
- `cache_features.py`: computes DRFP once per unique reaction and writes level-specific split caches.
- `train.py`: trains one level/split run with AMP and checkpoints.
- `evaluate.py`: computes classification and EC hierarchy metrics.
- `common.py`, `ec.py`, `modeling.py`: shared paths, EC parsing, datasets, and training helpers.
- `run_split_benchmarks.py`, `tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`: full-run and tuning launchers.
- `default_hparams_theia_original.json`, `commands.txt`: default settings and runnable examples.

## 7. Minimal Edits Outside `emulator_bench/`

- Required external edits: this `Plan.md` only.
- Why each edit is unavoidable: the adaptation workflow requires a root plan artifact before implementation.
- Why the edit does not change baseline behavior: it is documentation only.

## 8. Exact Execution Plan

1. Invoke commands with `conda run -n mldb`.
2. Copy EMULaToR split files into the Theia baseline data directory.
3. Validate schemas and EC labels.
4. Cache DRFP fingerprints once per unique reaction and create split-specific cached tables.
5. Launch training for each discovered split and EC level using default Theia settings.
6. Evaluate test metrics and hierarchy metrics.
7. Aggregate completed test-run metrics with `python -m emulator_bench.aggregate_results`, validating saved prediction metrics and publishing per-run CSV, grouped CSV/JSON, and diagnostics reports.
8. Run a small smoke test on `cuda:0`.
9. Record outputs, metrics, cache locations, and reproduction commands.
