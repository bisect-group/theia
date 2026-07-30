# Theia EMULaToR Bench

This bench retrains Theia on EMULaToR enzyme retrieval splits without modifying Theia's original source files.

## Inputs

Each split file must contain:

- `rxn_smiles`: reaction SMILES
- `ec_number`: EC label
- `uniprot_date`: preserved if present, not used by the model

The source split tree is copied from:

```text
~/github/EMULaToR/data/processed/datasets/enzyme_retrieval_dataset
```

to:

```text
~/github/EMULaToR/data/processed/baselines/theia
```

## Features And Cache

The model uses Theia's DRFP reaction fingerprints:

- `radius=2`
- `n_folded_length=10240`
- `root_central_atom=False`
- `include_hydrogens=True`

Fingerprints are cached once per unique reaction at:

```text
~/github/EMULaToR/data/processed/baselines/theia/embeddings/reactions/drfp_r2_10240_h1_root0
```

Each split also gets level-specific cached tables under:

```text
<split_dir>/theia_cache/<ec1|ec12|ec123|ec1234>/{train,val,test}.parquet
```

## Model And Training

The bench reuses `theia.ml.MLPClassifier`:

- input size `10240`
- hidden size `1664`
- output size equal to the split/level class count
- CrossEntropyLoss
- Adam with learning rate `0.001`
- ExponentialLR with gamma `0.9`
- batch size `64`
- max `50` epochs
- early stopping patience `5`, min improvement `0.001`

Bench-only speed and reliability additions:

- AMP: bf16 on Ampere-or-newer CUDA, otherwise fp16
- TF32 enabled for CUDA matmul/cuDNN
- atomic feature writes
- `checkpoint_last.pt` resume
- completed-run skipping via `metrics.json` and `predictions.csv`
- Rich progress displays

## Metrics

Theia is a classification model, so the bench reports:

- accuracy
- macro F1
- weighted F1
- MCC
- confusion matrix
- EC hierarchy accuracy for L1-L4

Partial EC labels match any valid predicted class under the known prefix. For example, `1.-.-.-` can match any predicted EC beginning with `1`, and `2.7.4.-` can match any predicted EC beginning with `2.7.4`.

Accepted numeric EC syntax:

- L1: `1-7`
- L2: `1-99`
- L3: `1-99`
- L4: `1-999`

## Commands

Use the `mldb` environment.

Copy and validate splits:

```bash
conda run -n mldb python emulator_bench/dataset_adapter.py \
  --source_dataset_dir /home/adhil/github/EMULaToR/data/processed/datasets/enzyme_retrieval_dataset \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/theia
```

Cache features:

```bash
conda run -n mldb python emulator_bench/cache_features.py \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/theia \
  --embeddings_dir /home/adhil/github/EMULaToR/data/processed/baselines/theia/embeddings \
  --num_workers 8 \
  --batch_size 1024
```

`--num_workers` controls parallel DRFP encoder processes. The parent process writes cache files atomically, so killed runs remain restartable.

Run all default split benchmarks:

```bash
conda run -n mldb python emulator_bench/run_split_benchmarks.py \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/theia \
  --embeddings_dir /home/adhil/github/EMULaToR/data/processed/baselines/theia/embeddings \
  --cache_num_workers 8 \
  --cache_batch_size 1024 \
  --gpus 0 1 2 3 \
  --runs_per_gpu 2
```

Each training subprocess is isolated with `CUDA_VISIBLE_DEVICES=<physical_gpu>` and receives `--device cuda:0`, so the visible device inside the subprocess is always local `cuda:0`.

Small smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n mldb python emulator_bench/run_split_benchmarks.py \
  --source_dataset_dir /home/adhil/github/EMULaToR/data/processed/datasets/enzyme_retrieval_dataset \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/theia_smoke \
  --embeddings_dir /home/adhil/github/EMULaToR/data/processed/baselines/theia_smoke/embeddings \
  --copy_splits \
  --levels ec1 \
  --split_groups random_splits \
  --max_rows 128 \
  --device cuda:0
```

Optuna tuning is optional. Default retraining does not use Optuna.

## Aggregate Metrics

Aggregate all completed Theia test runs:

```bash
conda run -n mldb python -m emulator_bench.aggregate_results \
  --base-dir /home/adhil/github/EMULaToR/data/processed/baselines/theia \
  --output-dir /home/adhil/github/EMULaToR/data/processed/baselines/theia/aggregated_metrics
```

The command validates prediction-derived classification and hierarchy metrics,
then writes `per_run_metrics.csv`, `summary_metrics.csv`,
`summary_metrics.json`, and `diagnostics.json`. Add `--strict` to fail after
writing reports when any discovered run is incomplete or invalid.
