# src/08_test_experts.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERT_NAMES = [
    "econ_left_authoritarian",
    "econ_left_libertarian",
    "econ_right_authoritarian",
    "econ_right_libertarian",
]
BASE_CONDITION = "base"

_DTYPE_MAP: dict[str, Any] = (
    {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    if _HAS_TORCH
    else {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}
)

# matches "Answer on the following scale: ..." sentence at end of persona prompts.
_SCALE_INSTRUCTION_RE = re.compile(
    r"\s*Answer on the following scale[^.]*\.",
    re.IGNORECASE,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === DATACLASSES ===

@dataclass
class ExpertCondition:
    name: str                     # "base" or one of EXPERT_NAMES
    adapter_path: Optional[Path]  # None for base condition
    target_quadrant: Optional[str]


@dataclass
class GenerationConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float


# === DATA LOADING ===

def load_json_or_jsonl(path: Path) -> Any:
    """Load a file that may be a single JSON object or a JSONL sequence."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    records = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}") from exc
    return records


def load_statements(path: Path) -> list[dict]:
    data = load_json_or_jsonl(path)
    if isinstance(data, list):
        statements = data
    elif isinstance(data, dict):
        if "statements" not in data:
            raise ValueError(f"method1/2 file missing 'statements' key: {path}")
        statements = data["statements"]
    else:
        raise ValueError(f"unexpected format in {path}")

    for i, s in enumerate(statements):
        for key in ("id", "text", "axis"):
            if key not in s:
                raise ValueError(f"statement {i} missing required key '{key}' in {path}")

    log.info("loaded %d statements from %s", len(statements), path.name)
    return statements


def load_personas(path: Path) -> dict[str, dict]:
    data = load_json_or_jsonl(path)
    if not isinstance(data, dict) or "counter_persona_prompts" not in data:
        raise ValueError(f"personas file missing 'counter_persona_prompts' key: {path}")
    personas = data["counter_persona_prompts"]
    log.info("loaded %d counter-personas from %s", len(personas), path.name)
    return personas


def load_questions(path: Path) -> list[dict]:
    data = load_json_or_jsonl(path)
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        if "questions" not in data:
            raise ValueError(f"method3 file missing 'questions' key: {path}")
        questions = data["questions"]
    else:
        raise ValueError(f"unexpected format in {path}")

    for i, q in enumerate(questions):
        for key in ("key", "question", "options"):
            if key not in q:
                raise ValueError(f"question {i} missing required key '{key}' in {path}")

    log.info("loaded %d questions from %s", len(questions), path.name)
    return questions


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# === MODEL LOADING ===

def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: Optional[Path],
    dtype: torch.dtype,
    device: str,
) -> tuple[Any, AutoTokenizer]:
    """
    Load base model + optional PEFT adapter, and tokenizer.
    Reloads the base model from scratch for each condition so adapter state
    never leaks between conditions.
    Logic:
        Loads tokenizer, aliases pad→eos for Mistral, loads base model at
        requested dtype, moves to device, sets eval mode, then wraps with
        PeftModel if an adapter path is given.
    """
    log.info("loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        # mistral has no pad token; alias to eos without resizing embeddings
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info("loading base model: %s  dtype=%s", base_model_name, dtype)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype)
    model = model.to(device)
    model.eval()

    if adapter_path is not None:
        log.info("loading adapter: %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

    return model, tokenizer


# === STEERING VECTORS ===

def load_steering_vector(path: Path, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Load a steering vector from a .pt file, normalising to unit norm.

    Handles the following storage formats:
      - raw tensor
      - dict with key 'vector'
      - dict with key 'final_vector'
      - dict with nested 'final_vectors.logistic_regression'

    Logic:
        Loads the tensor, extracts from whatever dict structure is present,
        flattens to 1-D, casts to target dtype/device, and normalises.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(raw, torch.Tensor):
        vec = raw
    elif isinstance(raw, dict):
        if "vector" in raw:
            vec = raw["vector"]
        elif "final_vector" in raw:
            vec = raw["final_vector"]
        elif "final_vectors" in raw:
            fv = raw["final_vectors"]
            if isinstance(fv, torch.Tensor):
                vec = fv
            elif isinstance(fv, dict):
                if "logistic_regression" in fv:
                    vec = fv["logistic_regression"]
                else:
                    raise ValueError(
                        f"ambiguous 'final_vectors' in {path}; available keys: {list(fv)}"
                    )
            else:
                raise ValueError(
                    f"unexpected type for 'final_vectors' in {path}: {type(fv)}"
                )
        else:
            raise ValueError(
                f"cannot locate vector in {path}; "
                f"expected key 'vector', 'final_vector', or 'final_vectors'. "
                f"available keys: {list(raw)}"
            )
    else:
        raise ValueError(f"unexpected type in {path}: {type(raw)}")

    if vec.dim() > 1:
        vec = vec.squeeze()
    if vec.dim() != 1:
        raise ValueError(f"expected 1-D vector in {path}, got shape {vec.shape}")

    vec  = vec.to(dtype=dtype, device=device)
    norm = vec.norm()
    if norm < 1e-8:
        raise ValueError(f"steering vector in {path} has near-zero norm")
    return vec / norm


# === PROJECTION ===

def quadrant_from_coordinates(econ: float, social: float) -> str:
    """
    Map PCT coordinates to a quadrant name.

    Convention (matching the Political Compass):
      econ < 0, social < 0  → econ_left_libertarian
      econ < 0, social >= 0 → econ_left_authoritarian
      econ >= 0, social < 0 → econ_right_libertarian
      econ >= 0, social >= 0→ econ_right_authoritarian
    """
    if econ < 0 and social < 0:
        return "econ_left_libertarian"
    if econ < 0 and social >= 0:
        return "econ_left_authoritarian"
    if econ >= 0 and social < 0:
        return "econ_right_libertarian"
    return "econ_right_authoritarian"


def compute_pct_projection(
    model: Any,
    tokenizer: AutoTokenizer,
    text: str,
    econ_vector: torch.Tensor,
    social_vector: torch.Tensor,
    projection_layer: int,
    device: str,
    *,
    prefix_text: Optional[str] = None,
) -> dict:
    """
    Project text into PCT coordinate space via hidden-state dot products.

    Logic:
        Tokenises text, runs a forward pass with output_hidden_states=True,
        extracts hidden states at projection_layer (0=embeddings, 1=layer1, ...),
        mean-pools over non-padding tokens, normalises the pooled vector, then
        dots with the unit-norm econ and social steering vectors.

        When prefix_text is supplied the forward pass still runs on the full
        text (so response token hidden states are contextualized by the
        statement via attention), but the mean pool is computed only over
        tokens that follow the prefix. This prevents the statement's own
        political signal from contaminating the projection coordinate.
        If the prefix consumes the entire sequence (edge case: very long
        statement + short max_length), the function falls back to pooling
        over all tokens rather than returning a zero vector.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len        = input_ids.shape[1]

    # build pool_mask: start from attention_mask, then zero out prefix tokens.
    pool_mask = attention_mask.clone()  # [1, seq_len]
    if prefix_text is not None:
        prefix_enc = tokenizer(
            prefix_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        n_prefix = prefix_enc["input_ids"].shape[1]
        n_mask   = min(n_prefix, seq_len)
        pool_mask[0, :n_mask] = 0
        # fallback: if no response tokens remain, pool over everything.
        if pool_mask.sum() == 0:
            pool_mask = attention_mask.clone()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states  # tuple: (embed, layer1, ..., layerN)
    if projection_layer >= len(hidden_states):
        raise ValueError(
            f"projection_layer={projection_layer} out of range; "
            f"model has {len(hidden_states)} hidden-state tensors (index 0 = embeddings)"
        )

    layer_out = hidden_states[projection_layer]  # [1, seq_len, hidden_dim]

    # mean pool over response tokens only (or all tokens if no prefix given).
    mask   = pool_mask.unsqueeze(-1).float()                                # [1, seq_len, 1]
    pooled = (layer_out.float() * mask).sum(dim=1) / mask.sum(dim=1)       # [1, hidden_dim]
    pooled = pooled.squeeze(0)                                              # [hidden_dim]

    pooled_norm = pooled.norm()
    if pooled_norm > 1e-8:
        pooled = pooled / pooled_norm

    pct_economic = pooled.dot(econ_vector.float()).item()
    pct_social   = pooled.dot(social_vector.float()).item()

    return {
        "pct_economic":      pct_economic,
        "pct_social":        pct_social,
        "predicted_quadrant": quadrant_from_coordinates(pct_economic, pct_social),
    }


# === DRY-RUN STUBS ===

def _dry_run_project(key: str, condition_name: str) -> dict:
    """
    Deterministic fake PCT coordinates for --dry-run.
    Biased toward the condition's own quadrant so match-rate stats are non-trivial.
    """
    h = int(hashlib.md5(f"{condition_name}:{key}".encode()).hexdigest()[:8], 16)
    econ_bias = 0.3 if "right" in condition_name else -0.3
    soc_bias  = 0.3 if "auth"  in condition_name else -0.3
    noise_e   = ((h & 0xFF) / 255.0 - 0.5) * 0.4
    noise_s   = (((h >> 8) & 0xFF) / 255.0 - 0.5) * 0.4
    e = max(-1.0, min(1.0, econ_bias + noise_e))
    s = max(-1.0, min(1.0, soc_bias  + noise_s))
    return {
        "pct_economic":       round(e, 6),
        "pct_social":         round(s, 6),
        "predicted_quadrant": quadrant_from_coordinates(e, s),
    }


def _dry_run_score_options(options: list[str]) -> list[dict]:
    """Stub NLL scores — first option always wins."""
    return [
        {
            "option_index":       i,
            "option_label":       chr(ord("A") + i),
            "option_text":        opt,
            "scored_option_text": " " + opt,
            "total_nll":          float(i + 1),
            "avg_nll":            float(i + 1) / max(len(opt.split()), 1),
            "num_option_tokens":  len(opt.split()),
        }
        for i, opt in enumerate(options)
    ]


# === GENERATION ===

def strip_scale_instruction_from_persona(text: str) -> str:
    """Remove the Likert-scale instruction sentence from a counter-persona prompt."""
    return _SCALE_INSTRUCTION_RE.sub("", text).strip()


def generate_response(
    model: Any,
    tokenizer: AutoTokenizer,
    prompt: str,
    gen_config: GenerationConfig,
    device: str,
) -> str:
    """Generate a free-form continuation, returning only the newly generated text."""
    enc       = tokenizer(prompt, return_tensors="pt", truncation=False, padding=False)
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": gen_config.max_new_tokens,
        "do_sample":      gen_config.do_sample,
        "pad_token_id":   tokenizer.eos_token_id,
    }
    if gen_config.do_sample:
        gen_kwargs["temperature"] = gen_config.temperature
        gen_kwargs["top_p"]       = gen_config.top_p

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)

    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# === PROMPT BUILDERS ===

def build_method1_prompt(statement_text: str) -> str:
    return (
        "You are responding to a political attitude statement.\n\n"
        f"Statement:\n{statement_text}\n\n"
        "Write a concise 2-3 sentence response explaining your position."
    )


def build_method2_prompt(persona_text: str, statement_text: str) -> str:
    clean = strip_scale_instruction_from_persona(persona_text)
    return (
        f"{clean}\n\n"
        f"Statement:\n{statement_text}\n\n"
        "Write a concise 2-3 sentence response explaining your position."
    )


def build_method3_prompt(question: str, options: list[str]) -> str:
    labels        = [chr(ord("A") + i) for i in range(len(options))]
    options_block = "\n".join(f"{lbl}. {opt}" for lbl, opt in zip(labels, options))
    return (
        "Answer the following survey question by choosing exactly one option.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_block}\n\n"
        "Answer:"
    )


# === NLL SCORING ===

def score_options_by_avg_nll(
    model: Any,
    tokenizer: AutoTokenizer,
    prompt: str,
    options: list[str],
    device: str,
) -> list[dict]:
    """
    Score each option by average NLL conditioned on the prompt.
    Logic:
        Tokenises the prompt (with BOS) and each option continuation (without
        special tokens) separately, concatenates them, runs a forward pass, and
        computes cross-entropy only over the option token positions. Average NLL
        = total CE / number of option tokens. A leading space is prepended to
        each option to preserve word-boundary tokeniser behaviour.
    """
    prompt_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids = prompt_enc["input_ids"].to(device)  # [1, P]
    P          = prompt_ids.shape[1]

    results: list[dict] = []
    for i, option_text in enumerate(options):
        continuation = " " + option_text
        cont_enc     = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)
        cont_ids     = cont_enc["input_ids"].to(device)  # [1, C]
        C            = cont_ids.shape[1]

        if C == 0:
            log.warning("option %d tokenised to empty continuation — skipping", i)
            continue

        full_ids = torch.cat([prompt_ids, cont_ids], dim=1)  # [1, P+C]

        with torch.inference_mode():
            outputs = model(input_ids=full_ids)

        logits = outputs.logits  # [1, P+C, vocab]

        # logits at positions P-1 .. P+C-2 predict tokens P .. P+C-1
        option_logits  = logits[0, P - 1 : P + C - 1, :]  # [C, vocab]
        option_targets = full_ids[0, P : P + C]            # [C]

        total_nll = F.cross_entropy(option_logits, option_targets, reduction="sum").item()
        avg_nll   = total_nll / C

        results.append({
            "option_index":       i,
            "option_label":       chr(ord("A") + i),
            "option_text":        option_text,
            "scored_option_text": continuation,
            "total_nll":          total_nll,
            "avg_nll":            avg_nll,
            "num_option_tokens":  C,
        })

    return results


# === METHOD 1 ===

def run_method1(
    model: Any,
    tokenizer: AutoTokenizer,
    condition: ExpertCondition,
    statements: list[dict],
    econ_vector: Optional[torch.Tensor],
    social_vector: Optional[torch.Tensor],
    projection_layer: int,
    gen_config: GenerationConfig,
    output_path: Path,
    limit: Optional[int],
    device: str,
    proj_model: Any,
    proj_tokenizer: AutoTokenizer,
    dry_run: bool = False,
) -> list[dict]:
    """
    Method 1 — Representativeness.

    For each statement: generate a free-form response, then project
    (statement + response) into PCT space and record coordinates.
    """
    stmts   = statements[:limit] if limit else statements
    gen_meta = {
        "max_new_tokens": gen_config.max_new_tokens,
        "do_sample":      gen_config.do_sample,
        "temperature":    gen_config.temperature,
        "top_p":          gen_config.top_p,
    }
    log.info("[method1] expert=%s  statements=%d", condition.name, len(stmts))
    records: list[dict] = []

    for idx, stmt in enumerate(stmts):
        prompt = build_method1_prompt(stmt["text"])
        if dry_run:
            generated = "[DRY-RUN]"
            proj      = _dry_run_project(stmt["id"], condition.name)
            proj_text = f"{stmt['text']}\n\nResponse:\n{generated}"
        else:
            generated = generate_response(model, tokenizer, prompt, gen_config, device)
            proj_text = f"{stmt['text']}\n\nResponse:\n{generated}"
            proj      = compute_pct_projection(
                proj_model, proj_tokenizer, proj_text,
                econ_vector, social_vector, projection_layer, device,
                prefix_text=f"{stmt['text']}\n\nResponse:\n",
            )

        matches = None
        if condition.target_quadrant is not None:
            matches = proj["predicted_quadrant"] == condition.target_quadrant

        record = {
            "method":                  "method1_representativeness",
            "expert":                  condition.name,
            "adapter_path":            str(condition.adapter_path) if condition.adapter_path else None,
            "target_quadrant":         condition.target_quadrant,
            "statement_id":            stmt["id"],
            "axis":                    stmt["axis"],
            "input_text":              stmt["text"],
            "prompt":                  prompt,
            "generated_response":      generated,
            "projection_text":         proj_text,
            "pct_economic":            proj["pct_economic"],
            "pct_social":              proj["pct_social"],
            "predicted_quadrant":      proj["predicted_quadrant"],
            "matches_target_quadrant": matches,
            "generation_metadata":     gen_meta,
        }
        records.append(record)

        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        if (idx + 1) % 10 == 0 or idx == len(stmts) - 1:
            log.info("[method1] expert=%s  %d/%d done", condition.name, idx + 1, len(stmts))

    return records


# === METHOD 2 ===

def run_method2(
    model: Any,
    tokenizer: AutoTokenizer,
    condition: ExpertCondition,
    statements: list[dict],
    personas: dict[str, dict],
    econ_vector: Optional[torch.Tensor],
    social_vector: Optional[torch.Tensor],
    projection_layer: int,
    gen_config: GenerationConfig,
    output_path: Path,
    all_m1_records: list[dict],
    limit: Optional[int],
    device: str,
    warnings: list[str],
    proj_model: Any,
    proj_tokenizer: AutoTokenizer,
    dry_run: bool = False,
) -> None:
    """
    Method 2 — Inverse Steerability.

    Applies an adversarial counter-persona prompt before each statement and
    measures how far the projection shifts from the Method 1 baseline.
    Trained experts run only their designated adversarial persona; the base
    model runs all four counter-personas.
    """
    if condition.name == BASE_CONDITION:
        personas_to_run = list(personas.items())
    else:
        personas_to_run = [
            (pid, pd)
            for pid, pd in personas.items()
            if pd.get("target_adapter") == condition.name
        ]
        if not personas_to_run:
            log.warning(
                "[method2] no designated adversary found for expert=%s — skipping",
                condition.name,
            )
            return

    m1_index: dict[str, dict] = {
        r["statement_id"]: r
        for r in all_m1_records
        if r.get("expert") == condition.name
    }
    if not m1_index:
        msg = (
            f"[method2] no method1 baselines for expert={condition.name} "
            "— shift/delta fields will be null"
        )
        log.warning(msg)
        warnings.append(msg)

    gen_meta = {
        "max_new_tokens": gen_config.max_new_tokens,
        "do_sample":      gen_config.do_sample,
        "temperature":    gen_config.temperature,
        "top_p":          gen_config.top_p,
    }
    stmts = statements[:limit] if limit else statements
    total = len(personas_to_run) * len(stmts)
    done  = 0

    log.info(
        "[method2] expert=%s  personas=%d  statements=%d  total=%d",
        condition.name, len(personas_to_run), len(stmts), total,
    )

    for persona_id, persona_data in personas_to_run:
        is_designated = (
            condition.name != BASE_CONDITION
            and persona_data.get("target_adapter") == condition.name
        )

        for stmt in stmts:
            prompt = build_method2_prompt(persona_data["text"], stmt["text"])
            if dry_run:
                generated = "[DRY-RUN]"
                proj      = _dry_run_project(f"{persona_id}:{stmt['id']}", condition.name)
                proj_text = f"{stmt['text']}\n\nResponse:\n{generated}"
            else:
                generated = generate_response(model, tokenizer, prompt, gen_config, device)
                proj_text = f"{stmt['text']}\n\nResponse:\n{generated}"
                proj      = compute_pct_projection(
                    proj_model, proj_tokenizer, proj_text,
                    econ_vector, social_vector, projection_layer, device,
                    prefix_text=f"{stmt['text']}\n\nResponse:\n",
                )

            baseline = m1_index.get(stmt["id"])
            if baseline is not None:
                m1_econ      = baseline["pct_economic"]
                m1_soc       = baseline["pct_social"]
                m1_quad      = baseline["predicted_quadrant"]
                delta_econ   = proj["pct_economic"] - m1_econ
                delta_soc    = proj["pct_social"]   - m1_soc
                shift_mag    = math.sqrt(delta_econ ** 2 + delta_soc ** 2)
                quad_changed = proj["predicted_quadrant"] != m1_quad
            else:
                m1_econ = m1_soc = m1_quad = None
                delta_econ = delta_soc = shift_mag = quad_changed = None

            record = {
                "method":                              "method2_inverse_steerability",
                "expert":                              condition.name,
                "adapter_path":                        str(condition.adapter_path) if condition.adapter_path else None,
                "target_quadrant":                     condition.target_quadrant,
                "persona_id":                          persona_id,
                "persona_label":                       persona_data.get("label", persona_id),
                "persona_target_adapter":              persona_data.get("target_adapter", ""),
                "is_designated_adversary":             is_designated,
                "statement_id":                        stmt["id"],
                "axis":                                stmt["axis"],
                "input_text":                          stmt["text"],
                "prompt":                              prompt,
                "generated_response":                  generated,
                "projection_text":                     proj_text,
                "pct_economic":                        proj["pct_economic"],
                "pct_social":                          proj["pct_social"],
                "predicted_quadrant":                  proj["predicted_quadrant"],
                "method1_baseline_pct_economic":       m1_econ,
                "method1_baseline_pct_social":         m1_soc,
                "method1_baseline_predicted_quadrant": m1_quad,
                "delta_economic":                      delta_econ,
                "delta_social":                        delta_soc,
                "shift_magnitude":                     shift_mag,
                "quadrant_changed_from_method1":       quad_changed,
                "generation_metadata":                 gen_meta,
            }

            with output_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            done += 1
            if done % 20 == 0 or done == total:
                log.info("[method2] expert=%s  %d/%d done", condition.name, done, total)


# === METHOD 3 ===

def run_method3(
    model: Any,
    tokenizer: AutoTokenizer,
    condition: ExpertCondition,
    questions: list[dict],
    econ_vector: Optional[torch.Tensor],
    social_vector: Optional[torch.Tensor],
    projection_layer: int,
    output_path: Path,
    limit: Optional[int],
    device: str,
    warnings: list[str],
    proj_model: Any,
    proj_tokenizer: AutoTokenizer,
    dry_run: bool = False,
) -> None:
    """
    Method 3 — Consistency.

    For each survey question, selects the answer option with the lowest average
    NLL, then projects (question + selected answer) into PCT space.
    'Refused' options are excluded from scoring.
    """
    qs = questions[:limit] if limit else questions
    log.info("[method3] expert=%s  questions=%d", condition.name, len(qs))

    for idx, q in enumerate(qs):
        valid_options = [
            opt for opt in q["options"]
            if opt.strip().lower() != "refused"
        ]

        if not valid_options:
            msg = f"[method3] question '{q['key']}' has no valid options — skipping"
            log.warning(msg)
            warnings.append(msg)
            continue

        prompt = build_method3_prompt(q["question"], valid_options)
        scored = (
            _dry_run_score_options(valid_options)
            if dry_run
            else score_options_by_avg_nll(model, tokenizer, prompt, valid_options, device)
        )

        if not scored:
            msg = f"[method3] question '{q['key']}' produced no scored options — skipping"
            log.warning(msg)
            warnings.append(msg)
            continue

        best        = min(scored, key=lambda x: x["avg_nll"])
        picked_text = best["option_text"]
        proj_text   = f"{q['question']}\n\nAnswer:\n{picked_text}"
        proj        = (
            _dry_run_project(q["key"], condition.name)
            if dry_run
            else compute_pct_projection(
                proj_model, proj_tokenizer, proj_text,
                econ_vector, social_vector, projection_layer, device,
                prefix_text=f"{q['question']}\n\nAnswer:\n",
            )
        )

        matches = None
        if condition.target_quadrant is not None:
            matches = proj["predicted_quadrant"] == condition.target_quadrant

        record = {
            "method":                  "method3_consistency",
            "expert":                  condition.name,
            "adapter_path":            str(condition.adapter_path) if condition.adapter_path else None,
            "target_quadrant":         condition.target_quadrant,
            "question_key":            q["key"],
            "question":                q["question"],
            "coarse_topic":            q.get("coarse_topic", ""),
            "fine_topics":             q.get("fine_topics", []),
            "axis_relevance":          q.get("axis_relevance", ""),
            "wave":                    q.get("wave", ""),
            "prompt":                  prompt,
            "options_scored":          scored,
            "picked_option_index":     best["option_index"],
            "picked_option_label":     best["option_label"],
            "picked_response":         picked_text,
            "picked_avg_nll":          best["avg_nll"],
            "picked_total_nll":        best["total_nll"],
            "projection_text":         proj_text,
            "pct_economic":            proj["pct_economic"],
            "pct_social":              proj["pct_social"],
            "predicted_quadrant":      proj["predicted_quadrant"],
            "matches_target_quadrant": matches,
        }

        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        if (idx + 1) % 20 == 0 or idx == len(qs) - 1:
            log.info("[method3] expert=%s  %d/%d done", condition.name, idx + 1, len(qs))


# === SUMMARIES ===

def _safe_mean(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None and not math.isnan(float(v))]
    return statistics.mean(vals) if vals else None


def _safe_std(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None and not math.isnan(float(v))]
    return statistics.stdev(vals) if len(vals) >= 2 else None


def _safe_median(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None and not math.isnan(float(v))]
    return statistics.median(vals) if vals else None


def compute_method1_summary(records: list[dict]) -> dict:
    by_expert: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_expert[r["expert"]].append(r)

    summaries: dict[str, dict] = {}
    for expert, recs in by_expert.items():
        econ_vals   = [r["pct_economic"] for r in recs]
        soc_vals    = [r["pct_social"]   for r in recs]
        target_quad = recs[0].get("target_quadrant")

        c_e = _safe_mean(econ_vals) or 0.0
        c_s = _safe_mean(soc_vals)  or 0.0

        match_vals = [
            r["matches_target_quadrant"] for r in recs
            if r["matches_target_quadrant"] is not None
        ]
        econ_recs = [r for r in recs if r.get("axis") == "economic"]
        soc_recs  = [r for r in recs if r.get("axis") == "social"]

        summaries[expert] = {
            "n_records":                            len(recs),
            "target_quadrant":                      target_quad,
            "mean_pct_economic":                    _safe_mean(econ_vals),
            "mean_pct_social":                      _safe_mean(soc_vals),
            "std_pct_economic":                     _safe_std(econ_vals),
            "std_pct_social":                       _safe_std(soc_vals),
            "centroid_quadrant":                    quadrant_from_coordinates(c_e, c_s),
            "quadrant_match_rate":                  (sum(match_vals) / len(match_vals)) if match_vals else None,
            "economic_statement_mean_pct_economic": _safe_mean([r["pct_economic"] for r in econ_recs]),
            "social_statement_mean_pct_social":     _safe_mean([r["pct_social"]   for r in soc_recs]),
        }

    return summaries


def compute_method2_summary(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["expert"], r["persona_id"])].append(r)

    # Pre-compute base-model shift per persona for resistance calculation
    base_shifts: dict[str, list[float]] = defaultdict(list)
    for (expert, persona_id), recs in groups.items():
        if expert == BASE_CONDITION:
            base_shifts[persona_id].extend(
                r["shift_magnitude"] for r in recs if r.get("shift_magnitude") is not None
            )

    summaries: list[dict] = []
    for (expert, persona_id), recs in sorted(groups.items()):
        sample    = recs[0]
        econ_vals = [r["pct_economic"] for r in recs]
        soc_vals  = [r["pct_social"]   for r in recs]
        d_econ    = [r["delta_economic"]  for r in recs if r.get("delta_economic")  is not None]
        d_soc     = [r["delta_social"]    for r in recs if r.get("delta_social")    is not None]
        shifts    = [r["shift_magnitude"] for r in recs if r.get("shift_magnitude") is not None]
        flips     = [
            r["quadrant_changed_from_method1"] for r in recs
            if r.get("quadrant_changed_from_method1") is not None
        ]

        base_mean         = _safe_mean(list(base_shifts.get(persona_id, [])))
        expert_mean_shift = _safe_mean(shifts)
        resistance        = (
            base_mean - expert_mean_shift
            if base_mean is not None and expert_mean_shift is not None
            else None
        )

        summaries.append({
            "expert":                  expert,
            "persona_id":              persona_id,
            "persona_label":           sample.get("persona_label", ""),
            "persona_target_adapter":  sample.get("persona_target_adapter", ""),
            "is_designated_adversary": sample.get("is_designated_adversary", False),
            "n_records":               len(recs),
            "mean_pct_economic":       _safe_mean(econ_vals),
            "mean_pct_social":         _safe_mean(soc_vals),
            "mean_delta_economic":     _safe_mean(d_econ),
            "mean_delta_social":       _safe_mean(d_soc),
            "mean_shift_magnitude":    expert_mean_shift,
            "median_shift_magnitude":  _safe_median(shifts),
            "quadrant_flip_rate":      (sum(flips) / len(flips)) if flips else None,
            "resistance_vs_base":      resistance,
        })

    return summaries


def compute_method3_summary(records: list[dict]) -> dict:
    by_expert: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_expert[r["expert"]].append(r)

    expert_summaries: dict[str, dict] = {}
    topic_summaries:  list[dict]      = []

    for expert, recs in by_expert.items():
        econ_vals   = [r["pct_economic"] for r in recs]
        soc_vals    = [r["pct_social"]   for r in recs]
        target_quad = recs[0].get("target_quadrant")
        c_e = _safe_mean(econ_vals) or 0.0
        c_s = _safe_mean(soc_vals)  or 0.0

        match_vals = [
            r["matches_target_quadrant"] for r in recs
            if r["matches_target_quadrant"] is not None
        ]

        expert_summaries[expert] = {
            "n_records":           len(recs),
            "target_quadrant":     target_quad,
            "mean_pct_economic":   _safe_mean(econ_vals),
            "mean_pct_social":     _safe_mean(soc_vals),
            "std_pct_economic":    _safe_std(econ_vals),
            "std_pct_social":      _safe_std(soc_vals),
            "centroid_quadrant":   quadrant_from_coordinates(c_e, c_s),
            "quadrant_match_rate": (sum(match_vals) / len(match_vals)) if match_vals else None,
        }

        by_topic: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_topic[r.get("coarse_topic", "unknown")].append(r)

        for topic, t_recs in sorted(by_topic.items()):
            t_econ = [r["pct_economic"] for r in t_recs]
            t_soc  = [r["pct_social"]   for r in t_recs]
            t_e    = _safe_mean(t_econ) or 0.0
            t_s    = _safe_mean(t_soc)  or 0.0
            topic_summaries.append({
                "expert":            expert,
                "coarse_topic":      topic,
                "n_questions":       len(t_recs),
                "mean_pct_economic": _safe_mean(t_econ),
                "mean_pct_social":   _safe_mean(t_soc),
                "std_pct_economic":  _safe_std(t_econ),
                "std_pct_social":    _safe_std(t_soc),
                "centroid_quadrant": quadrant_from_coordinates(t_e, t_s),
            })

    return {"by_expert": expert_summaries, "by_expert_topic": topic_summaries}


def compute_combined_summary(
    m1_summary: dict,
    m2_summary: list[dict],
    m3_by_expert: dict,
    all_conditions: list[ExpertCondition],
    warnings: list[str],
) -> dict:
    # index method2 designated-adversary entries by expert name
    m2_desig: dict[str, dict] = {
        e["expert"]: e for e in m2_summary if e.get("is_designated_adversary")
    }

    experts_out = []
    for cond in all_conditions:
        name = cond.name
        m1   = m1_summary.get(name, {})
        m3   = m3_by_expert.get(name, {})
        m2d  = m2_desig.get(name, {})

        experts_out.append({
            "expert":                          name,
            "target_quadrant":                 cond.target_quadrant,
            "method1_centroid_quadrant":        m1.get("centroid_quadrant"),
            "method1_mean_pct_economic":        m1.get("mean_pct_economic"),
            "method1_mean_pct_social":          m1.get("mean_pct_social"),
            "method1_quadrant_match_rate":      m1.get("quadrant_match_rate"),
            "method2_designated_mean_shift":    m2d.get("mean_shift_magnitude"),
            "method2_designated_flip_rate":     m2d.get("quadrant_flip_rate"),
            "method2_resistance_vs_base":       m2d.get("resistance_vs_base"),
            "method3_centroid_quadrant":        m3.get("centroid_quadrant"),
            "method3_mean_pct_economic":        m3.get("mean_pct_economic"),
            "method3_mean_pct_social":          m3.get("mean_pct_social"),
            "method3_quadrant_match_rate":      m3.get("quadrant_match_rate"),
        })

    return {"experts": experts_out, "warnings": warnings}


# === MAIN ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="evaluate trained LoRA experts using PCT projection methods"
    )

    # required
    p.add_argument("--model-name",             required=True)
    p.add_argument("--method12-path",          type=Path, required=True)
    p.add_argument("--method2-personas-path",  type=Path, required=True)
    p.add_argument("--method3-path",           type=Path, required=True)
    p.add_argument("--econ-vector-path",       type=Path, required=True)
    p.add_argument("--social-vector-path",     type=Path, required=True)
    p.add_argument("--output-dir",             type=Path, required=True)
    p.add_argument(
        "--projection-layer",
        type=int,
        default=20,
        help="hidden-state layer index for PCT projection (0=embeddings, N=layerN output); "
             "Mistral-7B has 33 states (0..32), default=20 (mid-network)",
    )

    # adapter paths — omit any to skip that expert
    p.add_argument("--adapter-econ-left-authoritarian",  type=Path, default=None)
    p.add_argument("--adapter-econ-left-libertarian",    type=Path, default=None)
    p.add_argument("--adapter-econ-right-authoritarian", type=Path, default=None)
    p.add_argument("--adapter-econ-right-libertarian",   type=Path, default=None)

    # generation
    p.add_argument("--max-new-tokens", type=int,   default=120)
    p.add_argument("--do-sample",      action="store_true", default=False)
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--top-p",          type=float, default=1.0)

    # scope
    p.add_argument(
        "--run-methods",
        nargs="+",
        choices=["method1", "method2", "method3"],
        default=["method1", "method2", "method3"],
    )
    p.add_argument("--limit",  type=int, default=None,
                   help="cap statements/questions per condition for smoke testing")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype",  choices=list(_DTYPE_MAP), default="bfloat16")
    p.add_argument("--append", action="store_true", default=False,
                   help="append to existing output JSONL files instead of overwriting; "
                        "summaries will include all records (old + new), so use with care")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="skip model/vector loading and use stub outputs — validates the full "
                        "pipeline (data loading, loops, file writing) without a GPU")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        log.info("DRY-RUN mode — model/vector loading skipped; outputs use stub coordinates")
    elif not _HAS_TORCH:
        raise SystemExit(
            "[error] torch/transformers/peft are not installed — cannot run without --dry-run.\n"
            "        Install dependencies or add --dry-run to validate the pipeline locally."
        )
    log.info("starting evaluation — methods=%s  limit=%s  dry_run=%s",
             args.run_methods, args.limit, args.dry_run)

    # validate input paths
    for attr, label in [
        ("method12_path",         "--method12-path"),
        ("method2_personas_path", "--method2-personas-path"),
        ("method3_path",          "--method3-path"),
        ("econ_vector_path",      "--econ-vector-path"),
        ("social_vector_path",    "--social-vector-path"),
    ]:
        p = getattr(args, attr)
        if not p.exists():
            raise FileNotFoundError(f"{label}: {p} not found")

    # build expert conditions
    adapter_map: dict[str, Optional[Path]] = {
        "econ_left_authoritarian":  args.adapter_econ_left_authoritarian,
        "econ_left_libertarian":    args.adapter_econ_left_libertarian,
        "econ_right_authoritarian": args.adapter_econ_right_authoritarian,
        "econ_right_libertarian":   args.adapter_econ_right_libertarian,
    }
    for name, path in adapter_map.items():
        if path is not None and not path.exists():
            raise FileNotFoundError(f"adapter path for {name} not found: {path}")

    base_condition    = ExpertCondition(name=BASE_CONDITION, adapter_path=None, target_quadrant=None)
    expert_conditions = [
        ExpertCondition(name=name, adapter_path=path, target_quadrant=name)
        for name, path in adapter_map.items()
        if path is not None
    ]
    all_conditions = [base_condition] + expert_conditions
    log.info(
        "conditions: base + %d experts: %s",
        len(expert_conditions), [c.name for c in expert_conditions],
    )

    # load datasets 
    statements = load_statements(args.method12_path)
    personas   = load_personas(args.method2_personas_path)
    questions  = load_questions(args.method3_path)

    for persona_id, pdata in personas.items():
        ta = pdata.get("target_adapter", "")
        if ta not in EXPERT_NAMES:
            raise ValueError(
                f"persona '{persona_id}' has target_adapter='{ta}' "
                f"which is not in {EXPERT_NAMES}"
            )

    # load steering vectors 
    dtype = _DTYPE_MAP[args.dtype]
    if args.dry_run:
        log.info("DRY-RUN: skipping steering vector load")
        econ_vector = social_vector = None
    else:
        log.info("loading steering vectors")
        econ_vector   = load_steering_vector(args.econ_vector_path,   args.device, dtype)
        social_vector = load_steering_vector(args.social_vector_path, args.device, dtype)
        log.info(
            "econ_vector shape=%s  social_vector shape=%s",
            econ_vector.shape, social_vector.shape,
        )

    # output setup 
    args.output_dir.mkdir(parents=True, exist_ok=True)

    m1_path = args.output_dir / "method1_representativeness.jsonl"
    m2_path = args.output_dir / "method2_inverse_steerability.jsonl"
    m3_path = args.output_dir / "method3_consistency.jsonl"

    if not args.append:
        for out_p in [m1_path, m2_path, m3_path]:
            out_p.write_text("", encoding="utf-8")
        log.info("cleared output JSONL files (pass --append to resume)")
    else:
        non_empty = [p for p in [m1_path, m2_path, m3_path] if p.exists() and p.stat().st_size > 0]
        if non_empty:
            log.warning(
                "--append: %d output file(s) already contain records; summaries will "
                "aggregate all existing + new records — use without --append to start fresh",
                len(non_empty),
            )

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    run_config = {
        "model_name":            args.model_name,
        "projection_model":      "base (fixed, no adapter)",
        "projection_layer":      args.projection_layer,
        "dtype":                 args.dtype,
        "device":                args.device,
        "run_methods":           args.run_methods,
        "limit":                 args.limit,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample":      args.do_sample,
            "temperature":    args.temperature,
            "top_p":          args.top_p,
        },
        "conditions": [
            {"name": c.name, "adapter_path": str(c.adapter_path) if c.adapter_path else None}
            for c in all_conditions
        ],
        "method12_path":         str(args.method12_path),
        "method2_personas_path": str(args.method2_personas_path),
        "method3_path":          str(args.method3_path),
        "econ_vector_path":      str(args.econ_vector_path),
        "social_vector_path":    str(args.social_vector_path),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("run config written → %s/run_config.json", args.output_dir)

    # load fixed base projection model 
    if args.dry_run:
        proj_model, proj_tokenizer = None, None
        log.info("DRY-RUN: skipping projection model load")
    else:
        log.info("loading fixed base projection model (no adapter)")
        proj_model, proj_tokenizer = load_model_and_tokenizer(
            args.model_name, None, dtype, args.device
        )

    warnings:       list[str]  = []
    all_m1_records: list[dict] = []

    # pre-load existing method1 records when method2 requested but method1 is not
    if "method2" in args.run_methods and "method1" not in args.run_methods:
        existing = _read_jsonl(m1_path)
        if existing:
            all_m1_records = existing
            log.info(
                "loaded %d existing method1 records for method2 baselines",
                len(existing),
            )
        else:
            msg = (
                "method2 selected without method1 and no existing method1 records "
                "found — shift fields will be null"
            )
            log.warning(msg)
            warnings.append(msg)

    # main evaluation loop 
    for condition in all_conditions:
        log.info("")
        log.info("=" * 60)
        log.info("condition: %s  adapter: %s", condition.name, condition.adapter_path)
        log.info("=" * 60)

        if args.dry_run:
            model, tokenizer = None, None
            log.info("DRY-RUN: skipping model load — condition: %s", condition.name)
        else:
            model, tokenizer = load_model_and_tokenizer(
                args.model_name, condition.adapter_path, dtype, args.device
            )

        if "method1" in args.run_methods:
            m1_recs = run_method1(
                model, tokenizer, condition, statements,
                econ_vector, social_vector, args.projection_layer,
                gen_config, m1_path, args.limit, args.device,
                proj_model, proj_tokenizer,
                dry_run=args.dry_run,
            )
            all_m1_records.extend(m1_recs)

        if "method2" in args.run_methods:
            run_method2(
                model, tokenizer, condition, statements, personas,
                econ_vector, social_vector, args.projection_layer,
                gen_config, m2_path, all_m1_records, args.limit, args.device, warnings,
                proj_model, proj_tokenizer,
                dry_run=args.dry_run,
            )

        if "method3" in args.run_methods:
            run_method3(
                model, tokenizer, condition, questions,
                econ_vector, social_vector, args.projection_layer,
                m3_path, args.limit, args.device, warnings,
                proj_model, proj_tokenizer,
                dry_run=args.dry_run,
            )

        if not args.dry_run:
            log.info("unloading generation model for condition: %s", condition.name)
            del model, tokenizer
            if args.device == "cuda":
                torch.cuda.empty_cache()

    if not args.dry_run:
        log.info("unloading projection model")
        del proj_model, proj_tokenizer
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # compute and write summaries 
    log.info("computing summaries")

    m1_records_all = _read_jsonl(m1_path)
    m2_records_all = _read_jsonl(m2_path)
    m3_records_all = _read_jsonl(m3_path)

    m1_summary = compute_method1_summary(m1_records_all) if m1_records_all else {}
    m2_summary = compute_method2_summary(m2_records_all) if m2_records_all else []
    m3_full    = (
        compute_method3_summary(m3_records_all)
        if m3_records_all
        else {"by_expert": {}, "by_expert_topic": []}
    )
    combined = compute_combined_summary(
        m1_summary, m2_summary, m3_full["by_expert"], all_conditions, warnings
    )

    (args.output_dir / "method1_summary.json").write_text(
        json.dumps(m1_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "method2_summary.json").write_text(
        json.dumps(m2_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "method3_summary.json").write_text(
        json.dumps(m3_full, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("all summaries written to %s", args.output_dir)
    log.info("evaluation complete — %d warnings", len(warnings))
    for w in warnings:
        log.warning(w)
    print(f"done — outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
