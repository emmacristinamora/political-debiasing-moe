#!/bin/bash
#SBATCH --job-name=test_experts
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=logs/test_experts_%j.out
#SBATCH --error=logs/test_experts_%j.err
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
mkdir -p data/experts/test-outputs

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

if [ ! -f "src/08_test_experts.py" ]; then
  echo "[error] src/08_test_experts.py not found — aborting"
  exit 1
fi

for quadrant in right_auth left_auth left_lib right_lib; do
  adapter_dir="data/experts/final-experts/${quadrant}"
  if [ ! -f "${adapter_dir}/adapter_config.json" ] || [ ! -f "${adapter_dir}/adapter_model.safetensors" ]; then
    echo "[error] final-experts/${quadrant} is missing adapter files — aborting"
    exit 1
  fi
done

for vec in economic_vectors social_vectors; do
  if [ ! -f "data/steering-vectors/vectors/${vec}.pt" ]; then
    echo "[error] data/steering-vectors/vectors/${vec}.pt not found — run 04_build_steering_vectors.py first"
    exit 1
  fi
done

for datafile in \
  "data/experts/test_experts/methode_1+2_data.jsonl" \
  "data/experts/test_experts/methode_2_counter_prompts.jsonl" \
  "data/experts/test_experts/methode_3_data.jsonl"
do
  if [ ! -s "${datafile}" ]; then
    echo "[error] ${datafile} is missing or empty — aborting"
    exit 1
  fi
done

echo "[info] all pre-flight checks passed"
echo "[info] start=$(date)"

# ── run evaluation ─────────────────────────────────────────────────────────────
# Adapter → condition name mapping:
#   final-experts/right_auth  →  --adapter-econ-right-authoritarian
#   final-experts/left_auth   →  --adapter-econ-left-authoritarian
#   final-experts/left_lib    →  --adapter-econ-left-libertarian
#   final-experts/right_lib   →  --adapter-econ-right-libertarian

python -u src/08_test_experts.py \
  --model-name                       mistralai/Mistral-7B-v0.1 \
  --method12-path                    data/experts/test_experts/methode_1+2_data.jsonl \
  --method2-personas-path            data/experts/test_experts/methode_2_counter_prompts.jsonl \
  --method3-path                     data/experts/test_experts/methode_3_data.jsonl \
  --econ-vector-path                 data/steering-vectors/vectors/economic_vectors.pt \
  --social-vector-path               data/steering-vectors/vectors/social_vectors.pt \
  --output-dir                       data/experts/test-outputs \
  --projection-layer                 20 \
  --adapter-econ-right-authoritarian data/experts/final-experts/right_auth \
  --adapter-econ-left-authoritarian  data/experts/final-experts/left_auth \
  --adapter-econ-left-libertarian    data/experts/final-experts/left_lib \
  --adapter-econ-right-libertarian   data/experts/final-experts/right_lib \
  --dtype                            bfloat16 \
  --device                           cuda \
  --max-new-tokens                   120 \
  --do-sample                         \
  --temperature                      0.9 \
  --top-p 0.9

echo ""
echo "[info] end=$(date)"
echo "[info] outputs in data/experts/test-outputs/"
