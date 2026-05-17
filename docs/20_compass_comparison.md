# Script 20 — Multi-Model Political Compass Comparison

## What this script does

Script 20 is the final evaluation layer of the pipeline. Its goal is to measure the **political bias** of several language models by placing them on a political compass and comparing their positions to our debiasing architecture (`run_moce`).

The core idea: given a set of politically charged prompts, we ask each model to respond 10 times, project every response onto the political compass using our steering vectors, and compute where each model sits on average.

---

## Pipeline

### Step 1 — Generation

For each of the 6 models, we generate **10 independent responses** per evaluation prompt (52 prompts total = 520 responses per model).

- **Standard models** use `model.generate()` with temperature sampling (`T=0.8`) so each of the 10 responses is distinct.
- **`run_moce`** calls `engine.run()` 10 times per prompt. The MoCE editor loop runs each time, mixing the four quadrant experts and applying recursive alignment corrections before producing the final text.

Results are written to disk **one prompt at a time** so the script is resumable if the job times out.

**Cache location:** `data/evaluation/compass_comparison/responses/<model_key>.jsonl`

### Step 2 — Projection

Every generated response (6 models × 52 prompts × 10 responses = **3,120 texts**) is encoded through **Mistral-7B** — the same model used to build the steering vectors — and projected onto the political compass.

Projection procedure for each text:
1. Tokenise and run a forward pass through Mistral-7B with `output_hidden_states=True`.
2. Take the output of **transformer block 20** (the encoding layer used throughout this project).
3. **Mean-pool** token representations under the attention mask.
4. Compute the **dot product** with the unit-normalised final-aggregate steering vector for each axis. Since the vector is unit-normalised, this equals cosine similarity.

This gives two scalars per response: an **economic score** and a **social score**.

### Step 3 — Aggregation

- **Per-prompt centroid**: mean of the 10 (economic, social) coordinate pairs for that prompt.
- **Global centroid per model**: mean of all 52 per-prompt centroids.

The global centroid is the model's estimated position on the political compass.

---

## Models evaluated

| Model | Type | HF login |
|---|---|---|
| `mistralai/Mistral-7B-v0.1` | Base LM, no instruction tuning | No |
| `run_moce` | Our debiasing architecture | No |
| `Qwen/Qwen2.5-7B-Instruct` | Instruction-tuned, ~7B | No |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Reasoning-distilled, ~7B | No |
| `meta-llama/Llama-3.1-8B-Instruct` | Instruction-tuned, ~8B | Yes |
| `google/gemma-2-9b-it` | Instruction-tuned, ~9B | Yes |

---

## Prompts

**File:** `data/evaluation/evaluation_prompts.jsonl`

52 prompts in JSONL format, each with fields `id`, `prompt_text`, `axis`, `source`.

- 20 held-out economic prompts (`holdout_eco_*`) — topics like housing, trade, taxation, labour rights
- 20 held-out social prompts (`holdout_soc_*`) — topics like free speech, euthanasia, surveillance, protest rights
- 12 smoke-test prompts (`baseline_neutral`, `econ_left`, `econ_right`, etc.) for sanity checking

These prompts were **never used** in training, PCT construction, steering vector extraction, or expert training — they are genuinely held-out.

---

## Projection details

| Parameter | Value |
|---|---|
| Projector model | `mistralai/Mistral-7B-v0.1` |
| Encoding layer | 20 |
| Vector method | `mean_difference` (final aggregate) |
| Axes | `economic`, `social` |
| Similarity | cosine (dot product with unit-normalised vector) |

The projection uses the same method as `src/18_compass_center.py` (compass center calibration) and `src/15_hein_project_compass.py` (legislator validation), ensuring consistency across the pipeline.

---

## Outputs

| File | Contents |
|---|---|
| `data/evaluation/compass_comparison/responses/<model>.jsonl` | Raw generated texts per prompt per model |
| `data/evaluation/compass_comparison/results.json` | Global centroids + full per-prompt breakdown |

### Structure of `results.json`

```json
{
  "n_prompts": 52,
  "n_responses": 10,
  "projection": {
    "model": "mistralai/Mistral-7B-v0.1",
    "layer": 20,
    "method": "mean_difference",
    "axes": ["economic", "social"]
  },
  "models": {
    "mistralai/Mistral-7B-v0.1": {
      "global_centroid": { "economic": -0.23, "social": 0.41 },
      "per_prompt": [
        {
          "prompt_id": "holdout_eco_1",
          "prompt_text": "...",
          "centroid": { "economic": -0.31, "social": 0.38 },
          "per_response": [
            { "economic": -0.28, "social": 0.35 },
            ...
          ]
        }
      ]
    },
    "run_moce": { ... },
    ...
  }
}
```

---

## Expected result

The key question is: **does `run_moce` sit closer to the political center than the other models?**

We expect:
- **Mistral 7B base** to occupy a consistent position reflecting the political tendencies of its pretraining data.
- **Instruction-tuned models** (Qwen, Llama, Gemma, DeepSeek) to cluster somewhere based on their RLHF alignment — likely slightly left-libertarian, a known tendency of RLHF-trained models.
- **`run_moce`** to sit closer to the origin `(0, 0)` — or at least meaningfully closer to the center than the base model — demonstrating that the debiasing architecture successfully reduces political bias without collapsing to a trivial output.

A successful result looks like a compass plot where `run_moce`'s global centroid is visually closer to center than the other models, with the per-prompt variance being low enough to make the centroid a reliable estimate.

---

## Running

```bash
# full run
python src/20_compass_comparison.py

# debug (5 prompts, 3 responses)
python src/20_compass_comparison.py --limit 5 --n-responses 3

# resume after timeout (skips models whose cache is complete)
python src/20_compass_comparison.py

# re-project only (all generation caches already exist)
python src/20_compass_comparison.py --skip-generation
```

On the cluster:
```bash
sbatch batch/submit_20_compass_comparison.sh
```
