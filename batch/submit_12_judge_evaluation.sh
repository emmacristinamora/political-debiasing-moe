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
POLARITY_KEY="config/pct_eval_polarity.yaml"

# ── sanity checks ──────────────────────────────────────────────────────────────

if [ ! -f "src/12_judge_evaluation.py" ]; then
  echo "[error] src/12_judge_evaluation.py not found — aborting"
  exit 1
fi

if [ ! -f "$POLARITY_KEY" ]; then
  echo "[error] ${POLARITY_KEY} not found — aborting"
  exit 1
fi

# the judge consumes the bias-radius generations; stage 11 must have run first
if [ ! -s "$BIAS_OUTPUTS" ]; then
  echo "[error] ${BIAS_OUTPUTS} missing or empty — run submit_11_evaluation.sh first"
  exit 1
fi

echo "[info] all pre-flight checks passed"

# ── Test 1: independent stance scoring ─────────────────────────────────────────
# A local Llama instruct judge classifies each answer's stance toward its PCT
# statement; stances + polarity key -> independent compass position per system.
# Writes data/evaluation/judge_stance/.
#
# The default judge (meta-llama/Llama-3.1-8B-Instruct) is a gated HF model:
# the account behind HF_HOME / HF_TOKEN must have been granted access, or pass
# an accessible model via --judge-model.

echo "[info] running judge stance scoring"

python -u src/12_judge_evaluation.py stance \
  --inputs    "$BIAS_OUTPUTS" \
  --polarity  "$POLARITY_KEY" \
  --device    cuda

# ── Test 2: blind pairwise preference ──────────────────────────────────────────
# The Llama judge compares base vs moce answers head-to-head on neutrality and
# coherence, in both orderings. Writes data/evaluation/judge_pairwise/.

echo "[info] running judge pairwise preference (base vs moce)"

python -u src/12_judge_evaluation.py pairwise \
  --inputs    "$BIAS_OUTPUTS" \
  --baseline  base \
  --treatment moce \
  --device    cuda

echo ""
echo "[info] end=$(date)"
echo "[info] outputs in data/evaluation/judge_stance/ and data/evaluation/judge_pairwise/"
