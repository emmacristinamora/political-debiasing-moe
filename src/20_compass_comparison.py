# src/20_compass_comparison.py
#
# Final-layer multi-model political compass evaluation.
#
# For each candidate model, generates N_RESPONSES responses per evaluation
# prompt, then projects every response onto the political compass via the
# Mistral-7B steering vectors (layer 20, final-aggregate mean-difference
# direction, cosine similarity). Per-prompt centroids are computed from the
# N_RESPONSES projections; those centroids are then averaged into a single
# global centroid per model.
#
#   === MODELS EVALUATED ===
#
#   - mistralai/Mistral-7B-v0.1      base reference (no finetuning)
#   - run_moce                        debiasing architecture (09/10_moce)
#   - Qwen/Qwen2.5-7B-Instruct       Qwen 2.5 7B Instruct
#   - tiiuae/falcon-7b-instruct       Falcon 7B Instruct (no HF login required)
#   - additional models via --models
#
#   === PIPELINE ===
#
#   Step 1 — GENERATION
#       For each model (in order), load it, sample N_RESPONSES completions
#       for every evaluation prompt, and save the texts to a per-model JSONL
#       cache under data/evaluation/compass_comparison/responses/.
#       Each model is fully unloaded (GPU memory freed) before the next one
#       is loaded, so two large models are never resident simultaneously.
#       The cache is checked before generating, making the script resumable
#       if interrupted.
#
#   Step 2 — PROJECTION
#       Load Mistral-7B once as the fixed "projector" model. For every
#       cached response text, encode it at transformer layer 20, mean-pool
#       the token representations under the attention mask, and compute the
#       dot product with the unit-normalised final-aggregate economic and
#       social steering vectors — this dot product equals cosine similarity
#       since the steering vectors are unit-normalised. The two scalars are
#       the response's political compass coordinates.
#
#   Step 3 — AGGREGATION
#       Per prompt:  centroid = mean of N_RESPONSES (economic, social) pairs.
#       Per model:   centroid = mean of per-prompt centroids.
#
#   === OUTPUTS ===
#
#   data/evaluation/compass_comparison/responses/<model_key>.jsonl
#       One JSON object per prompt with the 10 generated texts.
#
#   data/evaluation/compass_comparison/results.json
#       Per-model global centroids and full per-prompt breakdown
#       (coordinates for each individual response + prompt centroid).


from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# === CONFIG ===

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
COMPONENTS_PATH = Path(__file__).resolve().parent / "09_moce_components.py"

DEFAULT_PROMPTS    = PROJECT_ROOT / "data" / "evaluation" / "evaluation_prompts.jsonl"
DEFAULT_VECTORS    = PROJECT_ROOT / "data" / "steering-vectors" / "vectors"
DEFAULT_CONFIG     = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_OUT_DIR    = PROJECT_ROOT / "data" / "evaluation" / "compass_comparison"

PROJECTOR_MODEL  = "mistralai/Mistral-7B-v0.1"
ENCODING_LAYER   = 20      # hidden-state index; hidden_states[LAYER+1] is block LAYER's output
AXES             = ("economic", "social")
VECTOR_METHOD    = "mean_difference"

DEFAULT_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    "run_moce",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
]

N_RESPONSES       = 10
MAX_NEW_TOKENS    = 300
GENERATION_TEMP   = 0.8
MAX_PROMPT_TOKENS = 256

DTYPE_MAP = {
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
}


# === HELPERS: CLI ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Script 20 — multi-model political compass centroid evaluation."
    )
    p.add_argument("--prompts",       type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--vectors-dir",   type=Path, default=DEFAULT_VECTORS)
    p.add_argument("--config",        type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-dir",       type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--models",        nargs="+", default=DEFAULT_MODELS,
                   help="Model IDs to evaluate. Use 'run_moce' for the debiasing architecture.")
    p.add_argument("--n-responses",   type=int,   default=N_RESPONSES)
    p.add_argument("--temperature",   type=float, default=GENERATION_TEMP)
    p.add_argument("--max-new-tokens",type=int,   default=MAX_NEW_TOKENS)
    p.add_argument("--dtype",         choices=list(DTYPE_MAP), default="float16")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--proj-batch",    type=int, default=8,
                   help="Number of texts per projector forward pass.")
    p.add_argument("--limit",         type=int, default=None,
                   help="Cap prompts for debugging (e.g. --limit 5).")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip generation; only re-project existing cached responses.")
    p.add_argument("--calibrated",    action="store_true",
                   help="Use calibrated router for run_moce.")
    p.add_argument("--router-checkpoint", type=Path, default=None)
    return p.parse_args()


# === HELPERS: IO ===

def load_prompts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load evaluation prompts from a JSONL file (one JSON object per line)."""
    if not path.is_file():
        raise FileNotFoundError(f"prompts file not found: {path}")
    prompts = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts[:limit] if limit is not None else prompts


def load_moce_raw_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_key(model_id: str) -> str:
    """Filesystem-safe identifier (replace slashes with double underscores)."""
    return model_id.replace("/", "__")


def responses_cache_path(out_dir: Path, model_id: str) -> Path:
    return out_dir / "responses" / f"{model_key(model_id)}.jsonl"


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# === HELPERS: PROJECTION ===

def load_projector(model_name: str, dtype: torch.dtype, device: str) -> tuple[Any, Any]:
    """Load Mistral-7B in eval mode as the fixed compass projector."""
    print(f"  loading projector: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    mdl.to(device).eval()
    return mdl, tok


def load_steering_vectors(
    vectors_dir: Path,
    method: str,
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Load the final-aggregate steering direction for each axis.

    The returned tensors are unit-normalised so a dot product with a pooled
    hidden state equals cosine similarity.
    """
    vectors: dict[str, torch.Tensor] = {}
    for axis in AXES:
        path = vectors_dir / f"{axis}_vectors.pt"
        if not path.is_file():
            raise FileNotFoundError(f"steering vector not found: {path}")
        data = torch.load(path, map_location="cpu")
        v = data["final_vectors"][method].to(torch.float32)
        vectors[axis] = (v / (v.norm() + 1e-12)).to(device)
        print(f"  loaded {axis} vector  shape={tuple(vectors[axis].shape)}")
    return vectors


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token representations under the attention mask (mirrors scripts 15 and 18)."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def project_texts_batch(
    texts: list[str],
    projector: Any,
    projector_tok: Any,
    steering: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
) -> dict[str, list[float]]:
    """
    Project a list of texts onto both compass axes.

    For each text:
      1. Tokenise and encode with Mistral-7B, capturing all hidden states.
      2. Take the output of transformer block ENCODING_LAYER (index LAYER+1).
      3. Mean-pool token representations using the attention mask.
      4. Dot-product with the unit-normalised steering vector per axis.
         Because the vector is unit-normalised this equals cosine similarity.

    Returns {axis: [score, ...]} with one float per text per axis.
    """
    all_scores: dict[str, list[float]] = {axis: [] for axis in AXES}

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = projector_tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
            add_special_tokens=True,
        )
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = projector(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states is a tuple of length n_layers+1; index LAYER+1 is the
        # output of block LAYER (index 0 is the embedding layer output).
        layer_out = out.hidden_states[ENCODING_LAYER + 1].float()
        pooled    = mean_pool(layer_out, mask)             # [B, hidden_dim]

        for axis, unit_vec in steering.items():
            scores = (pooled @ unit_vec).cpu().tolist()    # cosine similarity
            all_scores[axis].extend(scores)

    return all_scores


# === HELPERS: GENERATION — STANDARD MODELS ===

def load_standard_model(
    model_id: str,
    dtype: torch.dtype,
    device: str,
) -> tuple[Any, Any]:
    print(f"  loading generator: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # left-padding for batch generation (keeps generated tokens contiguous)
    tok.padding_side = "left"
    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    mdl.to(device).eval()
    return mdl, tok


def format_prompt(prompt_text: str, tokenizer: Any) -> str:
    """
    Apply the model's chat template for instruction-tuned models.
    Falls back to the raw prompt text for base models (e.g. Mistral-7B-v0.1)
    which have no chat_template set.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return prompt_text


def generate_responses_standard(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    n: int,
    temperature: float,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    """
    Sample n independent completions for one prompt.

    The prompt is formatted with the model's chat template (if present),
    then tokenised and fed to model.generate() with num_return_sequences=n
    and temperature sampling. Only the newly generated tokens are decoded;
    the prompt prefix is stripped.
    """
    formatted = format_prompt(prompt_text, tokenizer)
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
        add_special_tokens=True,
    )
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    prompt_len = ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    responses: list[str] = []
    for seq in out:
        new_tokens = seq[prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)
    return responses


def generate_standard(
    model_id: str,
    prompts: list[dict[str, Any]],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    """
    Load a standard HuggingFace model, generate N responses per prompt,
    unload the model, and return a list of per-prompt records.
    """
    model, tokenizer = load_standard_model(model_id, dtype, args.device)
    records: list[dict[str, Any]] = []

    for i, prompt in enumerate(prompts, 1):
        pid  = prompt["id"]
        text = prompt["prompt_text"]
        print(f"  [{i}/{len(prompts)}] {pid}")
        try:
            responses = generate_responses_standard(
                model, tokenizer, text,
                n=args.n_responses,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
        except Exception as exc:
            print(f"    WARNING: generation failed for {pid}: {exc}")
            responses = []
        records.append({
            "prompt_id":   pid,
            "prompt_text": text,
            "model_id":    model_id,
            "responses":   responses,
        })

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return records


# === HELPERS: GENERATION — MoCE ENGINE ===

def _load_moce_components() -> Any:
    """Load src/09_moce_components.py via importlib (name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("moce_components", COMPONENTS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {COMPONENTS_PATH}")
    moce = importlib.util.module_from_spec(spec)
    sys.modules["moce_components"] = moce
    spec.loader.exec_module(moce)
    return moce


def build_moce_engine(
    raw_cfg: dict[str, Any],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    """
    Construct and return the MoCEEngine from config.yaml's moce_inference block.

    The generation sub-block is patched to enable temperature sampling so that
    repeated engine.run() calls produce distinct outputs — the config default
    is do_sample=False (greedy), which would yield identical responses.

    Returns (engine, moce_module) so the caller can access MoCEResult fields.
    """
    moce = _load_moce_components()
    inf  = raw_cfg["moce_inference"]

    # Enable sampling; override temperature and token budget from CLI.
    inf["generation"]["do_sample"]      = True
    inf["generation"]["temperature"]    = args.temperature
    inf["generation"]["max_new_tokens"] = args.max_new_tokens

    it_cfg = inf["input_transformer"]
    sv_cfg = inf["steering_vectors"]
    ckpts  = inf["expert_checkpoints"]
    rcfg   = inf["router"]
    ecfg   = inf["editor"]
    gcfg   = inf["generation"]

    # Load base model (Mistral-7B with LoRA adapters applied by ExpertManager).
    base_name = inf["model"]["base_model"]
    print(f"  loading MoCE base model: {base_name}")
    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    try:
        mdl = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
    except TypeError:
        mdl = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
    mdl.to(inf["model"]["device"]).eval()

    # Construct component configs (mirrors src/10_run_moce.py build_engine).
    steering_config = moce.SteeringVectorConfig(
        economic_vector_path=Path(sv_cfg["economic_vector_path"]),
        social_vector_path=Path(sv_cfg["social_vector_path"]),
        vector_method=it_cfg["vector_method"],
        use_final_aggregated_vectors=bool(it_cfg["use_final_aggregated_vectors"]),
        selected_layers=list(it_cfg["selected_layers"]),
        pooling_method=it_cfg["pooling_method"],
        use_centering=bool(it_cfg["use_centering"]),
        neutral_reference_path=it_cfg.get("neutral_reference_path"),
    )

    router_config = moce.RouterConfig(
        use_calibrated_router=bool(args.calibrated),
        beta=float(rcfg["beta"]),
        temperature=float(rcfg["temperature"]),
        calibration_input_dim=int(it_cfg["calibration_input_dim"]),
        fallback_to_uniform_if_centered=bool(rcfg["fallback_to_uniform_if_centered"]),
        center_threshold=float(rcfg["center_threshold"]),
    )

    expert_config = moce.ExpertConfig(
        left_lib_checkpoint=Path(ckpts["left_lib_checkpoint"]),
        left_auth_checkpoint=Path(ckpts["left_auth_checkpoint"]),
        right_lib_checkpoint=Path(ckpts["right_lib_checkpoint"]),
        right_auth_checkpoint=Path(ckpts["right_auth_checkpoint"]),
    )

    editor_config = moce.EditorConfig(
        max_edit_steps=int(ecfg["max_edit_steps"]),
        use_recursive_editing=bool(ecfg["use_recursive_editing"]),
        initialize_from_router=bool(ecfg["initialize_from_router"]),
        correction_beta=float(ecfg["correction_beta"]),
        convergence_threshold=float(ecfg["convergence_threshold"]),
        initialization_mode=str(ecfg["initialization_mode"]),
        keep_edit_trace=False,
        stop_on_axis_proximity=bool(ecfg["stop_on_axis_proximity"]),
        axis_proximity_threshold=float(ecfg["axis_proximity_threshold"]),
    )

    generation_config = moce.GenerationConfig(
        max_new_tokens=int(gcfg["max_new_tokens"]),
        temperature=float(gcfg["temperature"]),
        do_sample=bool(gcfg["do_sample"]),
        top_p=float(gcfg.get("top_p", 1.0)),
        frequency_penalty=float(gcfg.get("frequency_penalty", 0.0)),
        no_repeat_ngram_size=int(gcfg.get("no_repeat_ngram_size", 0)),
    )

    engine = moce.MoCEEngine(
        model=mdl,
        tokenizer=tok,
        steering_config=steering_config,
        router_config=router_config,
        expert_config=expert_config,
        editor_config=editor_config,
        generation_config=generation_config,
    )

    if args.calibrated and args.router_checkpoint is not None:
        engine.router.load_calibration_checkpoint(args.router_checkpoint)

    return engine, moce


def generate_moce(
    raw_cfg: dict[str, Any],
    prompts: list[dict[str, Any]],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    """
    Build the MoCEEngine, call engine.run() N times per prompt, collect
    the final_text from each MoCEResult, unload, and return records.

    Each call to engine.run() is independent; with do_sample=True the editor's
    expert-mixing + decoding will produce different outputs on each call.
    """
    engine, _ = build_moce_engine(raw_cfg, args, dtype)
    records: list[dict[str, Any]] = []

    for i, prompt in enumerate(prompts, 1):
        pid  = prompt["id"]
        text = prompt["prompt_text"]
        print(f"  [{i}/{len(prompts)}] {pid}")
        responses: list[str] = []
        for _ in range(args.n_responses):
            try:
                result = engine.run(text)
                responses.append(result.final_text)
            except Exception as exc:
                print(f"    WARNING: MoCE run failed for {pid}: {exc}")
                responses.append("")
        records.append({
            "prompt_id":   pid,
            "prompt_text": text,
            "model_id":    "run_moce",
            "responses":   responses,
        })

    del engine
    torch.cuda.empty_cache()
    gc.collect()
    return records


# === STEP 1: GENERATION ===

def run_generation_phase(
    prompts: list[dict[str, Any]],
    args: argparse.Namespace,
    dtype: torch.dtype,
    raw_cfg: dict[str, Any],
) -> None:
    """
    Generate and cache responses for every model.

    For each model:
      - Check if a cache file already exists (skip if so).
      - If model_id == "run_moce": build the MoCE engine and call engine.run().
      - Otherwise: load a standard HuggingFace model and call model.generate().
      - Save the per-prompt records to a JSONL cache file.
      - Explicitly free GPU memory before loading the next model.
    """
    print("\n=== STEP 1: GENERATION ===")

    for model_id in args.models:
        cache = responses_cache_path(args.out_dir, model_id)
        if cache.is_file():
            existing = load_jsonl(cache)
            if len(existing) >= len(prompts):
                print(f"[{model_id}] cache complete ({len(existing)} prompts) — skipping")
                continue
            print(f"[{model_id}] cache incomplete ({len(existing)}/{len(prompts)}) — regenerating")

        print(f"\n[{model_id}] generating {args.n_responses} responses × {len(prompts)} prompts")

        if model_id == "run_moce":
            records = generate_moce(raw_cfg, prompts, args, dtype)
        else:
            records = generate_standard(model_id, prompts, args, dtype)

        save_jsonl(records, cache)
        print(f"  saved {len(records)} records → {cache}")


# === STEP 2: PROJECTION ===

def project_all_responses(
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, list[dict[str, Any]]]:
    """
    Load Mistral-7B once as the projector, then project every cached response.

    For each model's cache file:
      - Collect all response texts.
      - Run project_texts_batch() in batches to get (economic, social) scores.
      - Reshape scores back into per-prompt, per-response structure.

    Returns:
        {model_id: [{"prompt_id": ..., "scores": [[econ, soc], ...10]}, ...]}
    """
    print("\n=== STEP 2: PROJECTION ===")
    print(f"projector model: {PROJECTOR_MODEL}  layer: {ENCODING_LAYER}  method: {VECTOR_METHOD}")

    projector, proj_tok = load_projector(PROJECTOR_MODEL, dtype, args.device)
    steering = load_steering_vectors(args.vectors_dir, VECTOR_METHOD, args.device)

    all_projections: dict[str, list[dict[str, Any]]] = {}

    for model_id in args.models:
        cache = responses_cache_path(args.out_dir, model_id)
        records = load_jsonl(cache)
        if not records:
            print(f"[{model_id}] no cache found — skipping projection")
            continue

        print(f"\n[{model_id}] projecting {len(records)} prompts × {args.n_responses} responses")

        # Flatten all response texts into a single list, keeping track of
        # how many responses each prompt contributed.
        flat_texts: list[str] = []
        lengths:    list[int] = []
        for rec in records:
            flat_texts.extend(rec["responses"])
            lengths.append(len(rec["responses"]))

        # Project the entire flat list in batches through Mistral-7B.
        flat_scores = project_texts_batch(
            flat_texts, projector, proj_tok, steering, args.device, args.proj_batch
        )
        # flat_scores = {"economic": [...], "social": [...]}

        # Reshape into per-prompt, per-response pairs.
        cursor = 0
        prompt_projections: list[dict[str, Any]] = []
        for rec, n in zip(records, lengths):
            per_response: list[dict[str, float]] = []
            for j in range(cursor, cursor + n):
                per_response.append({
                    "economic": flat_scores["economic"][j],
                    "social":   flat_scores["social"][j],
                })
            cursor += n
            prompt_projections.append({
                "prompt_id":   rec["prompt_id"],
                "prompt_text": rec["prompt_text"],
                "per_response": per_response,   # list of {economic, social}
            })

        all_projections[model_id] = prompt_projections

    del projector
    torch.cuda.empty_cache()
    gc.collect()

    return all_projections


# === STEP 3: AGGREGATION ===

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(
    all_projections: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """
    Compute per-prompt centroids and global per-model centroids.

    Per-prompt centroid: mean of N_RESPONSES (economic, social) coordinate pairs.
    Global centroid:     mean of per-prompt centroids across all prompts.

    Returns:
        {model_id: {
            "global_centroid":   {"economic": float, "social": float},
            "per_prompt": [{
                "prompt_id":   str,
                "prompt_text": str,
                "centroid":    {"economic": float, "social": float},
                "per_response": [{"economic": float, "social": float}, ...],
            }, ...],
        }}
    """
    print("\n=== STEP 3: AGGREGATION ===")
    results: dict[str, dict[str, Any]] = {}

    for model_id, prompt_projections in all_projections.items():
        per_prompt_out: list[dict[str, Any]] = []
        prompt_econ_centroids: list[float] = []
        prompt_soc_centroids:  list[float] = []

        for pp in prompt_projections:
            econ_scores = [r["economic"] for r in pp["per_response"]]
            soc_scores  = [r["social"]   for r in pp["per_response"]]

            # Centroid of the N_RESPONSES responses for this prompt.
            centroid_econ = _mean(econ_scores)
            centroid_soc  = _mean(soc_scores)

            prompt_econ_centroids.append(centroid_econ)
            prompt_soc_centroids.append(centroid_soc)

            per_prompt_out.append({
                "prompt_id":   pp["prompt_id"],
                "prompt_text": pp["prompt_text"],
                "centroid": {
                    "economic": centroid_econ,
                    "social":   centroid_soc,
                },
                "per_response": pp["per_response"],
            })

        # Global centroid: mean of all per-prompt centroids.
        global_centroid = {
            "economic": _mean(prompt_econ_centroids),
            "social":   _mean(prompt_soc_centroids),
        }

        results[model_id] = {
            "global_centroid": global_centroid,
            "per_prompt":      per_prompt_out,
        }

        print(
            f"  {model_id:<40} "
            f"economic={global_centroid['economic']:+.4f}  "
            f"social={global_centroid['social']:+.4f}"
        )

    return results


# === MAIN ===

def main() -> None:
    args  = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading prompts from {args.prompts}")
    prompts = load_prompts(args.prompts, args.limit)
    print(f"  {len(prompts)} prompts loaded")

    raw_cfg = load_moce_raw_config(args.config)

    # ------------------------------------------------------------------
    # Step 1: generation
    # ------------------------------------------------------------------
    if not args.skip_generation:
        run_generation_phase(prompts, args, dtype, raw_cfg)
    else:
        print("\n=== STEP 1: GENERATION (skipped — --skip-generation) ===")

    # ------------------------------------------------------------------
    # Step 2: projection (always runs, regardless of --skip-generation)
    # ------------------------------------------------------------------
    all_projections = project_all_responses(args, dtype)

    # ------------------------------------------------------------------
    # Step 3: aggregation
    # ------------------------------------------------------------------
    results = aggregate(all_projections)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = args.out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_prompts":   len(prompts),
                "n_responses": args.n_responses,
                "projection": {
                    "model":       PROJECTOR_MODEL,
                    "layer":       ENCODING_LAYER,
                    "method":      VECTOR_METHOD,
                    "axes":        list(AXES),
                },
                "models": results,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}")

    # ------------------------------------------------------------------
    # Print final summary table
    # ------------------------------------------------------------------
    print("\n=== FINAL SUMMARY — global centroid per model ===")
    print(f"  {'model':<40}  {'economic':>10}  {'social':>10}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*10}")
    for model_id, data in results.items():
        c = data["global_centroid"]
        print(f"  {model_id:<40}  {c['economic']:>+10.4f}  {c['social']:>+10.4f}")


if __name__ == "__main__":
    main()
