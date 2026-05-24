# Beyond One Political Voice: A Parliament-Inspired Mixture of Calibrated Experts for Political Debiasing

**Emma Mora · Riccardo Crespi · Stefana Chiriac**  
Bocconi University

> *Political bias in LLMs can appear through framing and emphasis, not only explicit stance. This makes neutrality hard to define: removing political content avoids the problem, while fine-tuning toward one neutral target creates another learned stance. Building on the two-dimensional Political Compass (PC) to locate a model's political stance, we instead define neutrality as the deliberate aggregation of opposing viewpoints. We embed the PC into Mistral-7B's hidden space and use it to guide Mixture-of-Calibrated-Experts (MoCE), a parliament-inspired architecture that balances these perspectives during generation. Diagnostic results show the PC directions are stable and externally meaningful on the economic axis, while MoCE improves over Mistral-7B in judge-based evaluation despite uneven expert specialization. By making a model's political position measurable and adjustable, MoCE points towards LLMs that engage contested issues fairly rather than avoiding them.*

---

## Overview

This repository is the full implementation accompanying the paper. It contains every component of the MoCE pipeline: the Political Compass embedding in Mistral-7B's representation space, the four quadrant-specific LoRA expert adapters, the heuristic and calibrated router, the iterative editor, and the evaluation suite.

**What MoCE does.** Given any prompt, MoCE (i) projects it onto the Political Compass using learned steering vectors, (ii) routes generation through a mixture of four politically-specialised experts weighted to counterbalance the measured bias, and (iii) iteratively refines the mixture until the output is near the compass centre. No base model weights are modified.

**Key results.**
- The economic steering vector is geometrically stable (5-fold CV AUC = 1.000) and externally meaningful: projected economic coordinates of 859 U.S. legislators correlate with DW-NOMINATE at Spearman ρ = +0.50 (p ≈ 10⁻⁴).
- Expert specialisation is asymmetric: only the right-libertarian adapter consistently reaches its target quadrant (95% quadrant-match rate); the others are constrained by corpus imbalance.
- MoCE is preferred over the unmodified base model by an LLM judge on neutrality (28% vs 8%), coherence (40% vs 18%), and relevance (35% vs 20%).
- MoCE reduces Mistral-7B's compass radius from 0.55 to 0.49, reaching the range of RLHF-aligned instruction-tuned models (Gemma-2, DeepSeek-R1, Llama-3.1).

For full methodology and results see the paper. For all numeric choices, thresholds, and design decisions see [`technical_context.md`](technical_context.md). For data provenance see [`data_statement.md`](data_statement.md).

---

## Repository Map

[`config/config.yaml`](config/config.yaml) is the single source of truth for every hyperparameter, path, and flag across all stages. [`technical_context.md`](technical_context.md) documents every design choice and threshold. [`data_statement.md`](data_statement.md) records data provenance.

### Steering Vectors (§ 2.1, Appendix A)

| Script | Description |
|---|---|
| `src/01_build_pairs.py` | 180 contrastive pairs — 30 seed statements × 3 templates × 2 axes |
| `src/02_validate_pairs.py` | Pair validation (length ratio, deduplication, format checks) |
| `src/03_extract_activations.py` | Mistral-7B hidden states at layers {8, 12, 16, 20, 24} — **GPU** |
| `src/04_build_steering_vectors.py` | Mean-difference and logistic-regression steering vectors |
| `src/13_steering_vector_geometry.py` | Tier 1: geometry checks (AUC, permutation test, k-fold CV) |
| `src/14_hein_build_dataset.py` | Hein corpus normalisation (congressional speeches) |
| `src/15_hein_project_compass.py` | Project Hein speeches onto the compass — **GPU** |
| `src/16_hein_dwnominate_analysis.py` | Tier 2: DW-NOMINATE correlation (Appendix A.2) |
| `src/17_steering_vector_robustness.py` | Tier 3: leave-one-out and cross-template invariance |

### Expert Adapters (§ 2.2, Appendix B)

| Script | Description |
|---|---|
| `data/experts/normalize_corpora.py` | Raw corpora (CSV, JSON, RDS) → canonical JSONL schema |
| `src/05_quadrant_datasets.py` | Corpus scoring, chunking, quadrant + topic labelling — **GPU** |
| `src/06_validate_experts_datasets.py` | Train/val splits with cell-capping and held-out topic/source |
| `src/07_train_experts.py` | LoRA fine-tuning of the four quadrant experts — **GPU** |
| `src/08_test_experts.py` | Three-method expert evaluation (representativeness, steerability, consistency) |

### MoCE Architecture (§ 2.3)

| Script | Description |
|---|---|
| `src/09_moce_components.py` | Full architecture: InputTransformer, Router, Editor, MoCEEngine |
| `src/10_run_moce.py` | End-to-end inference CLI (heuristic and calibrated routing) |
| `src/router_training/` | Calibrated router training pipeline (11 steps, see [`docs/router_calibration_pipeline.md`](docs/router_calibration_pipeline.md)) |

### Evaluation (§ 3, Appendix C)

| Script | Description |
|---|---|
| `src/11_moce_evaluation.py` | Automatic metrics: bias radius, quality, refusal/vagueness |
| `src/12_judge_evaluation.py` | LLM-as-judge: stance classification and pairwise comparison |
| `src/20_compass_comparison.py` | Multi-model compass centroid evaluation (§ 3.4) — **GPU** |
| `src/21_plot_compass_comparison.py` | Paper figure: compass positions of all evaluated models (Fig. 2) |

### Data

| Path | Contents |
|---|---|
| `data/steering-vectors/` | Activation cache and trained steering vector artifacts |
| `data/experts/raw/` | Normalised corpora, one JSONL per source (not tracked in git) |
| `data/experts/quadrant-pools/` | Scored and chunked pools per source |
| `data/experts/train-validate/` | Final train/val splits per expert |
| `data/experts/final-experts/` | Trained LoRA adapter checkpoints |
| `data/external/hein_dwnominate/` | Hein corpus and DW-NOMINATE projections |
| `data/evaluation/` | Judge outputs and compass comparison results |

### Infrastructure

| Path | Contents |
|---|---|
| `batch/` | SLURM submission scripts for every GPU stage |
| `notebooks/` | Exploratory analysis: steering vectors, expert datasets, evaluation |
| `docs/` | Training curves, router pipeline documentation, paper figures |
| `tests/` | pytest suite — no GPU required |

---

## Installation

**Requirements:** Python 3.12, conda, a single A100 or H200 GPU for training and inference.

```bash
conda create -n moce-proj python=3.12
conda activate moce-proj
pip install -r requirements.txt
```

The corpus normalisation step for House of Commons and EU Commission press releases additionally requires R (for RDS parsing):

```bash
# macOS
brew install r
# or follow https://cran.r-project.org for your platform
```

Model weights are downloaded automatically from Hugging Face on first use. Set your cache location:

```bash
export HF_HOME=/path/to/your/hf_cache
```

---

## Reproducing the Experiments

All hyperparameters, paths, and flags are in [`config/config.yaml`](config/config.yaml). SLURM submission scripts for cluster execution are in [`batch/`](batch/). The stages below correspond to sections of the paper.

### § 2.1 — Steering Vectors

```bash
# Build and validate contrastive pairs
python src/01_build_pairs.py       --config config/config.yaml
python src/02_validate_pairs.py    --config config/config.yaml

# Extract Mistral-7B activations (GPU)
python src/03_extract_activations.py --config config/config.yaml

# Build steering vectors
python src/04_build_steering_vectors.py --config config/config.yaml

# Validate — Tier 1 geometry (Appendix A.1)
python src/13_steering_vector_geometry.py --config config/config.yaml

# Validate — Tier 2 external ground truth (Appendix A.2)
python src/14_hein_build_dataset.py  --config config/config.yaml
python src/15_hein_project_compass.py --config config/config.yaml   # GPU
python src/16_hein_dwnominate_analysis.py --config config/config.yaml

# Validate — Tier 3 robustness (Appendix A.3)
python src/17_steering_vector_robustness.py --config config/config.yaml
```

### § 2.2 — Expert Datasets and Training

```bash
# Normalize raw corpora to canonical JSONL
python data/experts/normalize_corpora.py --config config/config.yaml

# Score corpora and assign quadrant + topic labels (GPU, ~9-10h for reddit_conservative)
python src/05_quadrant_datasets.py --config config/config.yaml --source allsides
python src/05_quadrant_datasets.py --config config/config.yaml --source reddit_liberal
python src/05_quadrant_datasets.py --config config/config.yaml --source reddit_conservative
python src/05_quadrant_datasets.py --config config/config.yaml --source ec_press
python src/05_quadrant_datasets.py --config config/config.yaml --source ire_press
python src/05_quadrant_datasets.py --config config/config.yaml --source uk_press
python src/05_quadrant_datasets.py --config config/config.yaml --source hoc
python src/05_quadrant_datasets.py --config config/config.yaml --source us_media
python src/05_quadrant_datasets.py --config config/config.yaml --source us_speeches
# Or run all sources in one SLURM job:
# sbatch batch/submit_05_quadrant_datasets.sh

# Build train/val splits
python src/06_validate_experts_datasets.py --config config/config.yaml

# Train four LoRA experts (GPU, ~4h per expert on H200)
python src/07_train_experts.py --config config/config.yaml --quadrant right_auth
python src/07_train_experts.py --config config/config.yaml --quadrant left_auth
python src/07_train_experts.py --config config/config.yaml --quadrant right_lib
python src/07_train_experts.py --config config/config.yaml --quadrant left_lib
# sbatch batch/submit_07_train_experts.sh

# Evaluate experts — three behavioural methods (Appendix B)
python src/08_test_experts.py --config config/config.yaml
```

### § 2.3 — MoCE Inference

```bash
# Single prompt, heuristic router
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompt "Should the government reduce economic inequality through taxation?"

# Batch prompts with output saved
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompts-file data/evaluation/prompts_for_evaluation.jsonl \
    --output-path data/smoke-test-outputs/moce_smoke_test.jsonl

# With calibrated router (requires running router_training/ first)
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompts-file data/evaluation/prompts_for_evaluation.jsonl \
    --output-path data/runs/calibrated.jsonl \
    --calibrated \
    --router-checkpoint data/router/checkpoints/calibrated_router.pt
```

See [`docs/router_calibration_pipeline.md`](docs/router_calibration_pipeline.md) for the full 11-step calibrated router training pipeline.

### § 3.3–3.4 — Evaluation

```bash
# LLM-as-judge: pairwise comparison (Appendix C.1)
python src/12_judge_evaluation.py pairwise \
    --config config/config.yaml \
    --responses-a data/smoke-test-outputs/moce_smoke_test.jsonl \
    --responses-b data/smoke-test-outputs/base_responses.jsonl

# Multi-model compass comparison (§ 3.4, Figure 2)
python src/20_compass_comparison.py --config config/config.yaml   # GPU
python src/21_plot_compass_comparison.py                           # produces docs/fig_compass_comparison.png
```

---

## Configuration

Every numeric hyperparameter, path, and flag  is in a single file: [`config/config.yaml`](config/config.yaml). Each stage has its own top-level block (`build_pairs`, `extract_activations`, `quadrant_dataset`, `train_experts`, `router_calibration`, `moce_inference`, …) so changes to one stage do not affect the others.

---

## Tests

```bash
pytest tests/
```

Covers the full `MoCEEngine` integration, all individual MoCE components, and the router-training pipeline. No GPU or model weights required.

---

