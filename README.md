# political-debiasing-moe

Mixture-of-Calibrated-Experts (MoCE) for debiasing political text generation.
A base causal LM (Mistral-7B) is paired with four LoRA experts — one per
quadrant of the political compass (left-libertarian, left-authoritarian,
right-libertarian, right-authoritarian) — and a router that mixes their
hidden states at decode time. The router emits a counterbalancing prior
that pushes the mixture *away* from the prompt's measured bias, and an
editor refines the mixture in compass space.

## What's implemented

End-to-end pipeline, GPU-ready:

- **Steering-vector pipeline** (`src/01_*` → `src/04_*`) — contrastive
  pairs on the economic and social axes, validation, activation extraction
  from Mistral-7B, and steering-vector construction.
- **Expert datasets and training** (`src/05_*` → `src/08_*`) — quadrant
  pools, dataset validation, LoRA training of the four quadrant experts,
  and a per-expert sanity test.
- **MoCE architecture** ([src/09_moce_components.py](src/09_moce_components.py)) —
  one self-contained module with all components:
  - `InputTransformer` projects prompts into compass space.
  - `Router` emits the heuristic counterbalancing prior π₀ and the
    optional calibrated policy π = softmax(log π₀ + δ(h)).
  - `ExpertManager` runs all four LoRA experts in dense mode.
  - `Editor` recursively re-mixes the experts' hidden states under
    alignment-driven corrections.
  - `MoCEEngine` owns end-to-end orchestration including token-by-token
    decode under the editor's converged mixture (see `MoCEEngine.run`).
- **End-to-end runner** ([src/10_run_moce.py](src/10_run_moce.py)) —
  CLI that loads the base model and adapters, builds `MoCEEngine`, and
  decodes one or many prompts. Heuristic routing is the default; calibrated
  routing is opt-in. See "Running the engine" below.
- **Calibrated-router training pipeline** ([src/router_training/](src/router_training/)) —
  prompt-set construction, feature/hidden-state extraction, forced-policy
  candidate-trace collection, candidate scoring, dataset splitting,
  trainer, evaluator, and a top-level pipeline driver. Outputs a checkpoint
  consumable by `--router-checkpoint` on the runner.

Not implemented yet:

- `src/11_evaluation.py` — placeholder; downstream metrics (bias-radius,
  refusal/vagueness, quality, robustness) are not wired into a CLI.

## Repo layout

```
config/config.yaml          single source of truth for every stage
src/01_…_08_…               steering vectors + expert datasets + training
src/09_moce_components.py   MoCE architecture (InputTransformer, Router,
                            ExpertManager, Editor, MoCEEngine)
src/10_run_moce.py          end-to-end inference runner (this README)
src/11_evaluation.py        stub
src/router_training/        calibrated-router training pipeline + CLI
batch/                      SLURM submission scripts for cluster runs
notebooks/                  exploratory analysis (steering vectors, experts)
tests/                      pytest suite (engine + router_training)
docs/                       calibrated-router pipeline docs
```

## Running the engine

`src/10_run_moce.py` is the single entrypoint for end-to-end MoCE
inference. It loads Mistral-7B + the four LoRA experts, builds a
`MoCEEngine`, and decodes prompts through
`transform → route → run experts → edit → decode`.

This is GPU-class work (~14 GB base weights in bf16, plus four LoRA
adapters and four parallel forward passes per generated token). Use a
single H100/A100, or submit it via the existing `batch/` SLURM pattern.

### Synopsis

```text
python src/10_run_moce.py
    --config <yaml>
    (--prompt <text> | --prompts-file <jsonl>)
    [--output-path <jsonl>]
    [--device cuda|cpu]
    [--calibrated --router-checkpoint <ckpt> [--calibration-input-dim N]]
```

### CLI reference

| Flag | Type | Required | Description |
| --- | --- | --- | --- |
| `--config` | path | yes | Path to `config.yaml`; must contain a `moce_inference:` block. |
| `--prompt` | string | one of these two | Single prompt string. Mutually exclusive with `--prompts-file`. |
| `--prompts-file` | path | one of these two | JSONL with one prompt per row: `{"prompt_text": "...", "id": "...optional..."}`. |
| `--output-path` | path | no | When provided, appends one JSON row per prompt with the full result. |
| `--device` | string | no | Overrides `moce_inference.model.device` (e.g. `cuda`, `cuda:0`, `cpu`). |
| `--calibrated` | flag | no | Switch the router to calibrated mode. Requires `--router-checkpoint`. |
| `--router-checkpoint` | path | iff `--calibrated` | Calibrated-router checkpoint produced by `src/router_training/`. |
| `--calibration-input-dim` | int | no | Override `moce_inference.input_transformer.calibration_input_dim`; must match the checkpoint. `--router-hidden-dim` is a deprecated alias. |
| `-h`, `--help` | flag | no | Print argparse help and exit. |

Contract checks fail loudly:

- exactly one of `--prompt` / `--prompts-file` must be given,
- `--calibrated` without `--router-checkpoint` is rejected,
- `--router-checkpoint` without `--calibrated` is rejected,
- missing `moce_inference` block or sub-key in `config.yaml` is rejected.

### `--prompts-file` JSONL schema

One JSON object per line:

```json
{"id": "q1", "prompt_text": "What role should government play in reducing economic inequality?"}
{"id": "q2", "prompt_text": "How should a society balance individual freedom and public safety?"}
```

`id` is optional; if present it is echoed in stdout and `--output-path` rows.

### `--output-path` JSONL schema

When `--output-path` is provided, one row per prompt is appended:

```json
{
  "id": "q1",
  "prompt_text": "...",
  "final_text": "...",
  "router_mode": "heuristic",
  "bias_magnitude": 0.34,
  "economic_score": -0.12,
  "social_score": 0.27,
  "quadrant_scores":   {"left_lib": 0.41, "left_auth": 0.18, "right_lib": 0.27, "right_auth": 0.14},
  "heuristic_prior":   {"left_lib": 0.12, "left_auth": 0.31, "right_lib": 0.22, "right_auth": 0.35},
  "calibrated_policy": {"left_lib": 0.12, "left_auth": 0.31, "right_lib": 0.22, "right_auth": 0.35},
  "final_alpha":       {"left_lib": 0.10, "left_auth": 0.33, "right_lib": 0.21, "right_auth": 0.36},
  "final_alignment":   {"left_lib": 0.08, "left_auth": 0.29, "right_lib": 0.18, "right_auth": 0.45},
  "num_edit_steps": 1,
  "stopped_early": false
}
```

In heuristic mode `calibrated_policy` mirrors `heuristic_prior` exactly.

### Example invocations

Single prompt, default heuristic routing:

```bash
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompt "What role should government play in reducing economic inequality?"
```

Batch with persisted rows:

```bash
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompts-file data/prompts/eval.jsonl \
    --output-path data/runs/heuristic_eval.jsonl
```

Calibrated routing (after running the router-training pipeline):

```bash
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompts-file data/prompts/eval.jsonl \
    --output-path data/runs/calibrated_eval.jsonl \
    --calibrated \
    --router-checkpoint data/router/checkpoints/best.pt
```

CPU smoke check (will load Mistral-7B; expect minutes per token):

```bash
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompt "..." \
    --device cpu
```

### `config.yaml` — `moce_inference` block

Self-contained block consumed only by `10_run_moce.py`:

```yaml
moce_inference:
  model:
    base_model: mistralai/Mistral-7B-v0.1
    dtype: bfloat16              # bfloat16 | float16 | float32
    device: cuda                 # overridable via --device

  steering_vectors:
    economic_vector_path: data/steering-vectors/vectors/economic_vectors.pt
    social_vector_path:   data/steering-vectors/vectors/social_vectors.pt

  input_transformer:
    vector_method: logistic_regression
    use_final_aggregated_vectors: true
    selected_layers: [8, 12, 16, 20, 24]
    pooling_method: mean
    use_centering: false
    neutral_reference_path: null
    calibration_input_dim: 4096  # overridable via --calibration-input-dim

  expert_checkpoints:
    left_lib_checkpoint:   data/experts/final-experts/left_lib
    left_auth_checkpoint:  data/experts/final-experts/left_auth
    right_lib_checkpoint:  data/experts/final-experts/right_lib
    right_auth_checkpoint: data/experts/final-experts/right_auth

  router:
    beta: 1.0                              # scales -beta * q_i in the prior
    temperature: 1.0                       # softmax temperature
    fallback_to_uniform_if_centered: true
    center_threshold: 0.05                 # bias_magnitude threshold for fallback

  editor:
    max_edit_steps: 1
    correction_beta: 1.0
    initialization_mode: router_policy     # router_policy | uniform
    use_recursive_editing: true
    initialize_from_router: true
    convergence_threshold: 1.0e-3

  generation:
    max_new_tokens: 256
    temperature: 0.7
    do_sample: false
    top_p: 1.0
```

Each adapter directory under `expert_checkpoints` must contain an
`adapter_config.json` plus PEFT weights. The four checkpoint paths are
mapped to the canonical quadrant order
`left_lib, left_auth, right_lib, right_auth`.

### Stdout output

For each prompt the runner prints a multi-line block:

```
========================================================================
id: q1
prompt:        What role should government play in reducing economic inequality?
final_text:    …generated answer…
prior:         left_lib=0.120, left_auth=0.310, right_lib=0.220, right_auth=0.350
policy:        left_lib=0.120, left_auth=0.310, right_lib=0.220, right_auth=0.350
final_alpha:   left_lib=0.105, left_auth=0.333, right_lib=0.210, right_auth=0.352
edit_steps:    1  stopped_early=False
```

## Other pipelines (brief pointers)

The numbered scripts under `src/` form an offline pipeline that produces
the artifacts consumed by `10_run_moce.py`:

- **Steering vectors** (`01` → `04`): produces `data/steering-vectors/vectors/{economic,social}_vectors.pt`.
- **Quadrant datasets** (`05` → `06`): produces the train/val splits used to fine-tune the experts.
- **Expert training** (`07`): LoRA-trains the four quadrant experts. Outputs land under `data/experts/`.
- **Expert testing** (`08`): per-expert sanity checks.
- **Calibrated router** ([src/router_training/](src/router_training/)): prompt-set construction → features → forced-policy traces → scored candidates → dataset splits → trainer → evaluator. See [docs/router_calibration_pipeline.md](docs/router_calibration_pipeline.md). The trainer's checkpoint is the file you pass to `--router-checkpoint` on the runner.

SLURM submission scripts for the cluster jobs are in [batch/](batch/).

## Tests

```bash
pytest tests/
```

The suite covers `MoCEEngine` integration, every individual MoCE
component, and the router-training pipeline. Tests do not require GPUs or
the Mistral-7B weights.

## Configuration

Every stage reads from a single [config/config.yaml](config/config.yaml).
Each stage has its own top-level block (`build_pairs`, `extract_activations`,
`quadrant_dataset`, `train_experts`, `router_calibration`, `moce_inference`, …)
so changes to one stage's hyperparameters do not affect the others.
