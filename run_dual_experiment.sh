#!/usr/bin/env bash
set -euo pipefail

cd /Users/ciaranwalsh/Documents/GitHub/realization-effect-project

MAXW=7
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p results/logs results/balance/blocks

ABS_LOG="results/logs/absolute_catchup_${TS}.log"
BAL_LOG="results/logs/balance_t1_n50_${TS}.log"
MERGE_LOG="results/logs/merge_after_dual_${TS}.log"

echo "[$(date)] Dual run starting (max_workers=${MAXW})"
echo "Absolute log: ${ABS_LOG}"
echo "Balance log:  ${BAL_LOG}"
echo "Merge log:    ${MERGE_LOG}"

# Balance model universe = all models with any absolute block data so far.
BAL_MODELS=()
while IFS= read -r _model; do
  if [[ -n "${_model}" ]]; then
    BAL_MODELS+=("${_model}")
  fi
done < <(python3 - <<'PY'
import csv
from pathlib import Path
models=set()
for fp in Path('results/blocks').glob('block__*.csv'):
    with fp.open(newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            if row.get('prompt_version')=='absolute' and row.get('model'):
                models.add(row['model'].strip())
for m in sorted(models):
    print(m)
PY
)

if (( ${#BAL_MODELS[@]} == 0 )); then
  echo "No balance models discovered from results/blocks; exiting."
  exit 1
fi

# Absolute catch-up gaps detected from results/blocks audit.
ABS_T1_MODELS=(
  "anthropic/claude-3.5-haiku"
  "google/gemini-3.1-pro-preview"
  "gpt-4.1-mini"
  "moonshotai/kimi-k2-thinking"
  "openai/gpt-5.4"
  "qwen/qwq-32b"
)
ABS_T05_MODELS=(
  "moonshotai/kimi-k2-thinking"
  "qwen/qwq-32b"
  "z-ai/glm-4.7"
)
ABS_T15_MODELS=(
  "google/gemini-3.1-pro-preview"
  "moonshotai/kimi-k2-thinking"
  "openai/gpt-5.4"
  "qwen/qwen3-235b-a22b"
  "qwen/qwen3-32b"
  "qwen/qwq-32b"
  "z-ai/glm-4.7"
  "z-ai/glm-5"
)

(
  set -euo pipefail
  echo "[$(date)] Starting ABSOLUTE catch-up"

  ./venv/bin/python run_experiment.py \
    --models "${ABS_T1_MODELS[@]}" \
    --temperatures 1.0 \
    --prompt-version absolute \
    --n-trials 100 \
    --max-workers "${MAXW}"

  ./venv/bin/python run_experiment.py \
    --models "${ABS_T05_MODELS[@]}" \
    --temperatures 0.5 \
    --prompt-version absolute \
    --n-trials 25 \
    --max-workers "${MAXW}"

  ./venv/bin/python run_experiment.py \
    --models "${ABS_T15_MODELS[@]}" \
    --temperatures 1.5 \
    --prompt-version absolute \
    --n-trials 25 \
    --max-workers "${MAXW}"

  echo "[$(date)] ABSOLUTE catch-up finished"
) >"${ABS_LOG}" 2>&1 &
ABS_PID=$!

(
  set -euo pipefail
  echo "[$(date)] Starting BALANCE t=1.0 n=50 across ${#BAL_MODELS[@]} model(s)"

  ./venv/bin/python run_experiment.py \
    --models "${BAL_MODELS[@]}" \
    --temperatures 1.0 \
    --prompt-version balance \
    --n-trials 50 \
    --max-workers "${MAXW}" \
    --output results/balance/results.csv

  echo "[$(date)] BALANCE run finished"
) >"${BAL_LOG}" 2>&1 &
BAL_PID=$!

echo "ABS_PID=${ABS_PID}"
echo "BAL_PID=${BAL_PID}"

ABS_STATUS=0
BAL_STATUS=0

if ! wait "${ABS_PID}"; then
  ABS_STATUS=$?
fi
if ! wait "${BAL_PID}"; then
  BAL_STATUS=$?
fi

{
  echo "[$(date)] Begin post-run merge"
  echo "ABS_STATUS=${ABS_STATUS}"
  echo "BAL_STATUS=${BAL_STATUS}"

  ./venv/bin/python reconcile_results.py

  echo "[$(date)] Post-run merge finished"
} >"${MERGE_LOG}" 2>&1

echo "[$(date)] Dual run complete"
