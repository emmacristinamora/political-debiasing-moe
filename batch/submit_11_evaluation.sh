#!/bin/bash
#SBATCH --job-name=moce_eval
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/moce_eval_%j.out
#SBATCH --error=logs/moce_eval_%j.err
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmamora2003@gmail.com

set -euo pipefail

echo "[info] job_id=${SLURM_JOB_ID:-no_slurm}"
echo "[info] node=$(hostname)"
echo "[info] start=$(date)"
echo "[info] partition=${SLURM_JOB_PARTITION:-unknown}"

REPO_ROOT="/home/3210604/projects/political-debiasing-moe"
cd "$REPO_ROOT"

mkdir -p logs
mkdir -p data/evaluation

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"

# ── inputs ─────────────────────────────────────────────────────────────────────

# single merged evaluation prompt set (charged + neutral); each record carries
# a "source" tag so the routing diagnostic still groups charged vs neutral.
EVAL_PROMPTS="data/evaluation/evaluation_prompts.jsonl"

# ── sanity checks ──────────────────────────────────────────────────────────────

if [ ! -f "src/11_moce_evaluation.py" ]; then
  echo "[error] src/11_moce_evaluation.py not found — aborting"
  exit 1
fi

if [ ! -f "src/09_moce_components.py" ]; then
  echo "[error] src/09_moce_components.py not found — aborting"
  exit 1
fi

if [ ! -f "config/config.yaml" ]; then
  echo "[error] config/config.yaml not found — aborting"
  exit 1
fi

if [ ! -f "$EVAL_PROMPTS" ]; then
  echo "[error] ${EVAL_PROMPTS} not found — aborting"
  exit 1
fi

for quadrant in left_lib left_auth right_lib right_auth; do
  if [ ! -d "data/experts/final-experts/${quadrant}" ]; then
    echo "[error] expert checkpoint data/experts/final-experts/${quadrant} not found — aborting"
    exit 1
  fi
done

for vec in economic_vectors social_vectors; do
  if [ ! -f "data/steering-vectors/vectors/${vec}.pt" ]; then
    echo "[error] data/steering-vectors/vectors/${vec}.pt not found — aborting"
    exit 1
  fi
done

echo "[info] all pre-flight checks passed"

# ── A: routing-geometry diagnostic ─────────────────────────────────────────────
# prefill only (no generation); profiles router decisiveness over the merged
# prompt set. The summary groups by the per-record "source" tag, so charged
# (methode_1+2) and neutral (smoke_test) stay distinguishable.
# Writes data/evaluation/routing_diagnostic/.

echo "[info] running routing-diagnostic"

python -u src/11_moce_evaluation.py routing-diagnostic \
  --config        config/config.yaml \
  --prompts-file  "$EVAL_PROMPTS" \
  --device        cuda

# ── B/E/F: output bias-radius across systems ───────────────────────────────────
# generates answers for base / moce / moce-single-step, re-encodes each to
# measure output bias-radius, and records quality metrics. This is the long
# stage. Writes data/evaluation/bias_radius/.

echo "[info] running bias-radius (systems: base, moce, moce-single-step)"

python -u src/11_moce_evaluation.py bias-radius \
  --config        config/config.yaml \
  --prompts-file  "$EVAL_PROMPTS" \
  --device        cuda

echo ""
echo "[info] end=$(date)"
echo "[info] outputs in data/evaluation/routing_diagnostic/ and data/evaluation/bias_radius/"
