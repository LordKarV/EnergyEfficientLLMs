#!/usr/bin/env bash
# Run the whole pipeline on a tiny random model to check that the code works.
# No GPU and no downloads needed. Takes well under a minute.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 tests/make_tiny_model.py
rm -rf outputs_smoke 2>/dev/null || mv outputs_smoke "outputs_smoke_old_$(date +%s)"

python3 scripts/01_generate_pool.py --config configs/smoke.yaml
python3 scripts/02_score_pool.py    --config configs/smoke.yaml
python3 scripts/03_select.py        --config configs/smoke.yaml
python3 scripts/04_train_lora.py    --config configs/smoke.yaml --arm qtrace
python3 scripts/05_evaluate.py      --config configs/smoke.yaml --model outputs_smoke/runs/qtrace_seed0/merged --tag qtrace_seed0
python3 scripts/06_summarize.py     --config configs/smoke.yaml
echo "SMOKE TEST PASSED"
