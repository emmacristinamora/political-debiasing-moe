#!/bin/bash
#SBATCH --job-name=score_candidates
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=logs/score_candidates_%j.out
#SBATCH --error=logs/score_candidates_%j.err
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

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"
echo "[info] gpu info:"
nvidia-smi || true

# ── sanity checks ─────────────────────────────────────────────────────────────

if [ ! -f "src/router_training/scorer.py" ]; then
  echo "[error] src/router_training/scorer.py not found — aborting"
  exit 1
fi

if [ ! -f "config/config.yaml" ]; then
  echo "[error] config/config.yaml not found — aborting"
  exit 1
fi

if [ ! -s "data/router/candidate_traces.jsonl" ]; then
  echo "[error] data/router/candidate_traces.jsonl missing or empty — run forced_policy_runner first"
  exit 1
fi

echo "[info] all pre-flight checks passed"

# ── filter empty final_text rows ──────────────────────────────────────────────
# The scorer raises ValueError on any trace with an empty final_text.
# A small number of prompts produce empty generations (model generates only
# special tokens); filter them out before scoring so the job doesn't crash.

echo "[info] filtering empty final_text rows from candidate_traces.jsonl..."
python3 - <<'PYEOF'
import json, sys
from pathlib import Path

src  = Path("data/router/candidate_traces.jsonl")
dst  = Path("data/router/candidate_traces_filtered.jsonl")

kept = skipped = 0
with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if t.get("final_text", "").strip():
            fout.write(line + "\n")
            kept += 1
        else:
            skipped += 1

print(f"[info] kept {kept} traces, skipped {skipped} empty/malformed rows",
      file=sys.stderr)
PYEOF

echo "[info] filter done — scoring filtered traces"
echo "[info] start=$(date)"

# ── score candidate traces ─────────────────────────────────────────────────────
# Loads Mistral-7B base (no adapters, no generation) and runs one forward pass
# per trace to get the PCT projection of final_text (bias_radius).
# Quality / refusal / vagueness are pure text heuristics — no GPU needed for those.
# Expected runtime: ~15–25 min for ~11 000 traces on H200.

python -u src/router_training/scorer.py \
  --config      config/config.yaml \
  --input-path  data/router/candidate_traces_filtered.jsonl \
  --output-path data/router/scored_traces.jsonl \
  --device      cuda

echo ""
echo "[info] end=$(date)"
echo "[info] scored traces written to data/router/scored_traces.jsonl"
