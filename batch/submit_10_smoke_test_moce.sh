#!/bin/bash
#SBATCH --job-name=smoke_moce
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/smoke_moce_%j.out
#SBATCH --error=logs/smoke_moce_%j.err
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
mkdir -p data/smoke-test-outputs

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"

# ── sanity checks ──────────────────────────────────────────────────────────────

if [ ! -f "src/10_run_moce.py" ]; then
  echo "[error] src/10_run_moce.py not found — aborting"
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

if [ ! -f "data/smoke_test_prompts.jsonl" ]; then
  echo "[error] data/smoke_test_prompts.jsonl not found — aborting"
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
echo "[info] running MoCE smoke test (heuristic routing, 12 prompts)"

# ── run MoCE inference ─────────────────────────────────────────────────────────

python -u src/10_run_moce.py \
  --config        config/config.yaml \
  --prompts-file  data/smoke_test_prompts.jsonl \
  --output-path   data/smoke-test-outputs/moce_smoke_test.jsonl \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42 \
  --device  cuda

echo ""
echo "[info] end=$(date)"
echo "[info] output saved to data/smoke-test-outputs/moce_smoke_test.jsonl"
