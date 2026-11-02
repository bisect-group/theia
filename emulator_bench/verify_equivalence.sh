#!/usr/bin/env bash
# Checks that the Theia wrapper matches the original code (CPU, fp32). Usage: verify_equivalence.sh
# Expected env: mldb (torch, drfp, rdkit, scikit-learn, pandas, pytest). The script does not create envs.
# Tests use 192 train, 64 val and 64 test real rows (tests/fixtures), seed 666. The original scripts/train.py needs
# pycm, shap and plotext only for reports, so the test stubs them.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
out=$(CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_original_equivalence.py -q -rs -p no:cacheprovider --color=no 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
echo "OK: wrapper matches the original code"
