#!/bin/bash
#SBATCH --job-name=router_candidates
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=logs/router_candidates_%j.out
#SBATCH --error=logs/router_candidates_%j.err
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
mkdir -p data/router

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"
echo "[info] gpu info:"
nvidia-smi || true

# ── sanity checks ─────────────────────────────────────────────────────────────

if [ ! -f "src/router_training/forced_policy_runner.py" ]; then
  echo "[error] src/router_training/forced_policy_runner.py not found — aborting"
  exit 1
fi

if [ ! -f "config/config.yaml" ]; then
  echo "[error] config/config.yaml not found — aborting"
  exit 1
fi

if [ ! -f "data/router/features.jsonl" ]; then
  echo "[error] data/router/features.jsonl not found — run features.py first"
  exit 1
fi

for quadrant in left_lib left_auth right_lib right_auth; do
  adapter_dir="data/experts/final-experts/${quadrant}"
  if [ ! -f "${adapter_dir}/adapter_config.json" ]; then
    echo "[error] final-experts/${quadrant}/adapter_config.json missing — aborting"
    exit 1
  fi
done

echo "[info] all pre-flight checks passed"
echo "[info] start=$(date)"

# ── runtime estimate ───────────────────────────────────────────────────────────
# 28 candidate policies per prompt × ~1.2 s/generation on H200 ≈ 34 s/prompt
#
# features.jsonl is source-sorted (first 500 rows = 92% left_lib, 0% right_auth)
# so --limit alone is badly skewed.  Use --stratify N to sample N prompts per
# quadrant (left_lib / left_auth / right_lib / right_auth) from the shuffled pool.
#
# --stratify 200  →  800 prompts  →  ~7.5 h   (22 400 traces, recommended)
# --stratify  50  →  200 prompts  →  ~1.9 h   (smoke test)
# --stratify 500  →  2000 prompts →  ~19 h    (generous; needs longer time limit)
#
# Adapter → quadrant mapping (overrides config checkpoint paths to final-experts):
#   --adapter-left-lib   data/experts/final-experts/left_lib
#   --adapter-left-auth  data/experts/final-experts/left_auth
#   --adapter-right-lib  data/experts/final-experts/right_lib
#   --adapter-right-auth data/experts/final-experts/right_auth

python -u src/router_training/forced_policy_runner.py \
  --config              config/config.yaml \
  --adapter-left-lib    data/experts/final-experts/left_lib \
  --adapter-left-auth   data/experts/final-experts/left_auth \
  --adapter-right-lib   data/experts/final-experts/right_lib \
  --adapter-right-auth  data/experts/final-experts/right_auth \
  --stratify            100 \
  --device              cuda

echo ""
echo "[info] end=$(date)"
echo "[info] traces written to data/router/candidate_traces.jsonl"
