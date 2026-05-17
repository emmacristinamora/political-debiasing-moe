#!/bin/bash
#SBATCH --job-name=compass_comparison
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=logs/compass_comparison_%j.out
#SBATCH --error=logs/compass_comparison_%j.err
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
mkdir -p data/evaluation/compass_comparison/responses

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Required for gated models (Llama 3.1, Gemma 2).
# Set your token in the cluster environment or uncomment and paste it here.
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[warn] HF_TOKEN is not set — Llama and Gemma downloads will fail if not already cached"
fi

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"
echo "[info] gpu info:"
nvidia-smi || true

# ── sanity checks ──────────────────────────────────────────────────────────────

if [ ! -f "src/20_compass_comparison.py" ]; then
  echo "[error] src/20_compass_comparison.py not found — aborting"
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

if [ ! -f "data/evaluation/evaluation_prompts.jsonl" ]; then
  echo "[error] data/evaluation/evaluation_prompts.jsonl not found — aborting"
  exit 1
fi

for vec in economic_vectors social_vectors; do
  if [ ! -f "data/steering-vectors/vectors/${vec}.pt" ]; then
    echo "[error] data/steering-vectors/vectors/${vec}.pt not found — aborting"
    exit 1
  fi
done

for quadrant in left_lib left_auth right_lib right_auth; do
  if [ ! -d "data/experts/final-experts/${quadrant}" ]; then
    echo "[error] expert checkpoint data/experts/final-experts/${quadrant} not found — aborting"
    exit 1
  fi
done

echo "[info] all pre-flight checks passed"

# ── run script 20 ──────────────────────────────────────────────────────────────
#
# Timing estimate (H100, 52 prompts, 10 responses each):
#   Mistral 7B base     ~6 min
#   run_moce            ~90 min   (sequential engine.run() calls with editor loop)
#   Qwen 2.5 7B         ~6 min
#   DeepSeek R1 Distill ~7 min
#   Llama 3.1 8B        ~7 min
#   Gemma 2 9B          ~8 min
#   projection phase    ~6 min
#   ─────────────────────────────
#   total               ~270 min  (8h wall time includes buffer)
#
# The script caches responses per model to
#   data/evaluation/compass_comparison/responses/<model_key>.jsonl
# so if the job times out it can be resumed with --skip-generation for
# models whose cache files are already complete.

echo ""
echo "[info] ===== Running script 20: compass comparison ====="

python -u src/20_compass_comparison.py \
  --prompts        data/evaluation/evaluation_prompts.jsonl \
  --vectors-dir    data/steering-vectors/vectors \
  --config         config/config.yaml \
  --out-dir        data/evaluation/compass_comparison \
  --n-responses    10 \
  --temperature    0.8 \
  --max-new-tokens 300 \
  --dtype          float16 \
  --device         cuda \
  --proj-batch     8

echo ""
echo "[info] ===== compass comparison done ====="
echo "[info] end=$(date)"
echo "[info] results written to data/evaluation/compass_comparison/results.json"
