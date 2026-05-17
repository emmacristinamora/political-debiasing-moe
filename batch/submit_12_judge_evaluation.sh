#!/bin/bash
#SBATCH --job-name=judge_eval
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:30:00
#SBATCH --output=logs/judge_eval_%j.out
#SBATCH --error=logs/judge_eval_%j.err
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

BIAS_OUTPUTS="data/evaluation/bias_radius/per_output.jsonl"

# ── sanity checks ──────────────────────────────────────────────────────────────

if [ ! -f "src/12_judge_evaluation.py" ]; then
  echo "[error] src/12_judge_evaluation.py not found — aborting"
  exit 1
fi

# the judge consumes the bias-radius generations; stage 11 must have run first
if [ ! -s "$BIAS_OUTPUTS" ]; then
  echo "[error] ${BIAS_OUTPUTS} missing or empty — run submit_11_evaluation.sh first"
  exit 1
fi

echo "[info] all pre-flight checks passed"

# ── External-Judge Pairwise Evaluation ─────────────────────────────────────────
# A local instruct judge compares base vs moce answers head-to-head on neutrality,
# coherence, and relevance, in both orderings. Writes data/evaluation/judge_pairwise/.

echo "[info] running external-judge pairwise evaluation (base vs moce)"

python -u src/12_judge_evaluation.py \
  --inputs      "$BIAS_OUTPUTS" \
  --baseline    base \
  --treatment   moce \
  --device      cuda \
  --judge-model Qwen/Qwen2.5-7B-Instruct

echo ""
echo "[info] end=$(date)"
echo "[info] outputs in data/evaluation/judge_pairwise/"
