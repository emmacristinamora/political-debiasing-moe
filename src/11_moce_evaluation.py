# src/11_evaluation.py

# MoCE evaluation suite. Each evaluation is a subcommand that runs a metric
# and writes its raw per-prompt records (JSONL) plus a summary (JSON) under
# data/evaluation/<metric>/. Visualization lives in notebooks/evaluation.ipynb,
# which reads these files. To add a metric: write a run_<metric>(args) function
# and register a subparser in parse_args().


# === IMPORTS ===

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS_PATH = PROJECT_ROOT / "src" / "09_moce_components.py"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluation")


# === DATACLASSES ===

@dataclass
class PromptRecord:
    """A single evaluation prompt with provenance for grouped reporting."""
    prompt_id: str
    text: str
    source: str               # file stem the prompt was loaded from
    axis: str | None = None   # "economic" / "social" when known, else None


# === COMPONENT + MODEL LOADING ===

def load_components_module() -> Any:
    """
    Load src/09_moce_components.py via importlib.

    The module name begins with a digit and cannot be imported with normal
    "import" syntax, so it is loaded explicitly by absolute path.
    """
    spec = importlib.util.spec_from_file_location("moce_components", COMPONENTS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load components module at {COMPONENTS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["moce_components"] = module
    spec.loader.exec_module(module)
    return module


def resolve_path(path_str: str) -> Path:
    """Resolve a config path string against the repo root when relative."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_dtype(name: str) -> Any:
    """Resolve a config dtype string into a torch.dtype."""
    import torch  # noqa: PLC0415

    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "float32":  torch.float32,
        "fp32":     torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(aliases)}")
    return aliases[name]


def load_inference_block(config_path: Path) -> dict[str, Any]:
    """
    Load config.yaml and return the moce_inference block.

    Logic:
        Reads the YAML file and validates that the sub-blocks needed for
        prompt encoding and heuristic routing are present, so misconfiguration
        surfaces here rather than deep inside component construction.
    """
    import yaml  # noqa: PLC0415

    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("moce_inference"), dict):
        raise ValueError(f"{config_path}: missing or non-mapping 'moce_inference' block")
    block = raw["moce_inference"]
    for key in ("model", "steering_vectors", "input_transformer", "router"):
        if not isinstance(block.get(key), dict):
            raise ValueError(f"{config_path}: moce_inference.{key} missing or not a mapping")
    return block


def load_model_and_tokenizer(base_model: str, dtype: Any, device: str) -> tuple[Any, Any]:
    """
    Load the base causal LM and tokenizer used for prompt encoding.

    Mirrors src/10_run_moce.py's loader so prefill geometry matches live
    MoCE inference (pad->eos alias, right padding, eval mode).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    log.info("loading tokenizer: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info("loading base model: %s  dtype=%s  device=%s", base_model, dtype, device)
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def build_steering_config(moce: Any, inference_cfg: dict[str, Any]) -> Any:
    """Construct a SteeringVectorConfig from the moce_inference config block."""
    it_cfg = inference_cfg["input_transformer"]
    sv = inference_cfg["steering_vectors"]
    return moce.SteeringVectorConfig(
        economic_vector_path=resolve_path(sv["economic_vector_path"]),
        social_vector_path=resolve_path(sv["social_vector_path"]),
        vector_method=it_cfg["vector_method"],
        use_final_aggregated_vectors=bool(it_cfg["use_final_aggregated_vectors"]),
        selected_layers=list(it_cfg["selected_layers"]),
        pooling_method=it_cfg["pooling_method"],
        use_centering=bool(it_cfg["use_centering"]),
        neutral_reference_path=it_cfg.get("neutral_reference_path"),
    )


def build_input_transformer(moce: Any, inference_cfg: dict[str, Any], model: Any, tokenizer: Any) -> Any:
    """Construct an InputTransformer from the moce_inference config block."""
    return moce.InputTransformer(
        model=model,
        tokenizer=tokenizer,
        steering_config=build_steering_config(moce, inference_cfg),
    )


def build_heuristic_router(moce: Any, inference_cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Construct a heuristic-mode Router; return (router, router_config)."""
    rcfg = inference_cfg["router"]
    it_cfg = inference_cfg["input_transformer"]
    router_config = moce.RouterConfig(
        use_calibrated_router=False,
        beta=float(rcfg["beta"]),
        temperature=float(rcfg["temperature"]),
        calibration_input_dim=int(it_cfg["calibration_input_dim"]),
        fallback_to_uniform_if_centered=bool(rcfg["fallback_to_uniform_if_centered"]),
        center_threshold=float(rcfg["center_threshold"]),
    )
    return moce.Router(router_config), router_config


# === IO HELPERS ===

def load_prompt_records(path: Path) -> list[PromptRecord]:
    """
    Load a prompt set from JSON or JSONL into PromptRecord objects.

    Logic:
        A pretty-printed single JSON document parses as a whole; a JSONL file
        does not, so a failed whole-file parse triggers line-by-line decoding.
        A JSON object is expected to hold prompts under a "statements" or
        "prompts" list. Each item must provide a text field ("text",
        "prompt_text", or "statement") and may carry "id"/"prompt_id" and
        "axis". The file stem becomes the source tag for grouped reporting.
    """
    if not path.is_file():
        raise FileNotFoundError(f"prompts file not found: {path}")
    source = path.stem
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"prompts file is empty: {path}")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = None  # not a single JSON document; decode as JSONL below

    if parsed is None:
        items: list[Any] = []
        for lineno, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {lineno}: invalid JSON") from exc
    elif isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = next(
            (parsed[key] for key in ("statements", "prompts") if isinstance(parsed.get(key), list)),
            None,
        )
        if items is None:
            raise ValueError(f"{path}: JSON object has no 'statements' or 'prompts' list")
    else:
        raise ValueError(f"{path}: unexpected top-level JSON type {type(parsed).__name__}")

    records: list[PromptRecord] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {index}: expected a JSON object")
        text = item.get("text") or item.get("prompt_text") or item.get("statement")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path} item {index}: missing a non-empty text field")
        raw_id = item.get("id", item.get("prompt_id", f"{source}_{index}"))
        axis = item.get("axis")
        records.append(
            PromptRecord(
                prompt_id=str(raw_id),
                text=text.strip(),
                source=source,
                axis=str(axis) if axis is not None else None,
            )
        )
    if not records:
        raise ValueError(f"{path}: no prompts found")
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, obj: Any) -> None:
    """Write a pretty-printed JSON summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# === STATISTICS HELPERS ===

def percentile(ordered: list[float], q: float) -> float:
    """Linear-interpolated q-quantile (q in [0, 1]) of an already-sorted list."""
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def percentile_summary(values: list[float]) -> dict[str, float]:
    """Mean and a min/p10/p25/median/p75/p90/max spread for a list of values."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean":   statistics.fmean(ordered),
        "min":    ordered[0],
        "p10":    percentile(ordered, 0.10),
        "p25":    percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "p75":    percentile(ordered, 0.75),
        "p90":    percentile(ordered, 0.90),
        "max":    ordered[-1],
    }


def normalized_entropy(weights: list[float]) -> float:
    """
    Shannon entropy of a distribution, normalized so 1.0 == uniform.

    A normalized entropy near 1.0 means the router is near-uniform (no
    routing signal); values toward 0.0 mean a peaked, decisive prior.
    """
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"prior weights must sum to a positive value; got {total}")
    probs = [w / total for w in weights]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return entropy / math.log(len(probs))


# === DIAGNOSTIC: ROUTING GEOMETRY ===

def build_diagnostic_row(
    record: PromptRecord,
    prompt_state: Any,
    prior: dict[str, float],
    router_config: Any,
) -> dict[str, Any]:
    """Assemble one per-prompt routing-geometry record."""
    weights = [float(v) for v in prior.values()]
    bias_magnitude = float(prompt_state.bias_magnitude)
    used_fallback = (
        bool(router_config.fallback_to_uniform_if_centered)
        and bias_magnitude < float(router_config.center_threshold)
    )
    return {
        "prompt_id": record.prompt_id,
        "source": record.source,
        "axis": record.axis,
        "prompt_text": record.text,
        "economic_score": float(prompt_state.economic_score),
        "social_score": float(prompt_state.social_score),
        "bias_magnitude": bias_magnitude,
        "quadrant_scores": {k: float(v) for k, v in prompt_state.quadrant_scores.items()},
        "heuristic_prior": {k: float(v) for k, v in prior.items()},
        "prior_entropy_norm": normalized_entropy(weights),
        "prior_max_weight": max(weights),
        "prior_top_quadrant": max(prior, key=prior.get),
        "used_center_fallback": used_fallback,
    }


def group_stats(rows: list[dict[str, Any]], center_threshold: float) -> dict[str, Any]:
    """Aggregate routing-geometry stats for one group of per-prompt rows."""
    count = len(rows)
    bias = [r["bias_magnitude"] for r in rows]
    below = sum(1 for b in bias if b < center_threshold)
    return {
        "n": count,
        "bias_magnitude": percentile_summary(bias),
        "prior_entropy_norm": percentile_summary([r["prior_entropy_norm"] for r in rows]),
        "prior_max_weight": percentile_summary([r["prior_max_weight"] for r in rows]),
        "fraction_below_center_threshold": below / count if count else 0.0,
    }


def summarize_routing(rows: list[dict[str, Any]], router_config: Any) -> dict[str, Any]:
    """
    Build the routing-geometry summary.

    Logic:
        Reports stats overall and grouped by source file and by political
        axis. The headline question is whether bias_magnitude has enough
        spread on charged prompts to drive a non-uniform prior, or whether
        the router collapses to uniform regardless of input.
    """
    center_threshold = float(router_config.center_threshold)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
        if row["axis"] is not None:
            by_axis[row["axis"]].append(row)
    return {
        "config": {
            "center_threshold": center_threshold,
            "beta": float(router_config.beta),
            "temperature": float(router_config.temperature),
        },
        "n_prompts": len(rows),
        "overall": group_stats(rows, center_threshold),
        "by_source": {k: group_stats(v, center_threshold) for k, v in sorted(by_source.items())},
        "by_axis": {k: group_stats(v, center_threshold) for k, v in sorted(by_axis.items())},
    }


def run_routing_diagnostic(args: argparse.Namespace) -> None:
    """
    Profile prompt geometry and the heuristic router across a prompt set.

    Logic:
        Loads the base model, builds InputTransformer + heuristic Router, and
        runs transform() on every prompt to obtain economic/social/quadrant
        scores and bias_magnitude, then builds the heuristic prior. Per-prompt
        geometry plus prior entropy and peak weight go to JSONL; a grouped
        summary goes to JSON. Together they answer whether the router has
        dynamic range or collapses to a near-uniform prior on all inputs.
    """
    inference_cfg = load_inference_block(args.config)
    model_cfg = inference_cfg["model"]
    device = args.device or str(model_cfg["device"])
    dtype = resolve_dtype(args.dtype or str(model_cfg["dtype"]))

    records: list[PromptRecord] = []
    for path in args.prompts_files:
        loaded = load_prompt_records(path)
        log.info("loaded %d prompts from %s", len(loaded), path.name)
        records.extend(loaded)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("no prompts to evaluate")

    moce = load_components_module()
    model, tokenizer = load_model_and_tokenizer(str(model_cfg["base_model"]), dtype, device)
    transformer = build_input_transformer(moce, inference_cfg, model, tokenizer)
    router, router_config = build_heuristic_router(moce, inference_cfg)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prompt_state = transformer.transform(record.text)
        prior = router.build_heuristic_prior(prompt_state)
        rows.append(build_diagnostic_row(record, prompt_state, prior, router_config))
        if index % 10 == 0 or index == len(records):
            log.info("transformed %d/%d prompts", index, len(records))

    per_prompt_path = args.output_dir / "per_prompt.jsonl"
    summary_path = args.output_dir / "summary.json"
    summary = summarize_routing(rows, router_config)
    write_jsonl(per_prompt_path, rows)
    save_json(summary_path, summary)

    overall = summary["overall"]
    log.info("wrote %s (%d rows)", per_prompt_path, len(rows))
    log.info("wrote %s", summary_path)
    log.info(
        "bias_magnitude: median=%.4f p90=%.4f | %.0f%% below center_threshold=%.3f",
        overall["bias_magnitude"]["median"],
        overall["bias_magnitude"]["p90"],
        100.0 * overall["fraction_below_center_threshold"],
        float(router_config.center_threshold),
    )


# === METRIC: BIAS RADIUS ===

# Output bias-radius across systems (plan items B / E / F). For each system
# the script generates an answer per prompt, re-encodes the answer with a
# pristine InputTransformer to measure its compass position, and records
# lightweight degeneration metrics. The three systems are:
#   base              raw base model, no experts / no steering
#   moce              full MoCE, multi-round recursive editing
#   moce-single-step  full MoCE, a single editing round
# moce vs moce-single-step isolates the value of iterating the editor.
BIAS_RADIUS_SYSTEMS = ("base", "moce", "moce-single-step")


def free_cuda() -> None:
    """Release a finished engine's GPU memory before loading the next."""
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_moce_engine(
    moce: Any,
    inference_cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    recursive_editing: bool,
) -> Any:
    """
    Construct a heuristic-mode MoCEEngine from the moce_inference block.

    recursive_editing toggles the editor between full recursive editing (up
    to editor.max_edit_steps) and a single-step pass. Mirrors the dataclass
    wiring in src/10_run_moce.py's build_engine so inference geometry matches.
    """
    it_cfg = inference_cfg["input_transformer"]
    rcfg = inference_cfg["router"]
    ckpts = inference_cfg["expert_checkpoints"]
    ecfg = inference_cfg["editor"]
    gcfg = inference_cfg["generation"]

    router_config = moce.RouterConfig(
        use_calibrated_router=False,
        beta=float(rcfg["beta"]),
        temperature=float(rcfg["temperature"]),
        calibration_input_dim=int(it_cfg["calibration_input_dim"]),
        fallback_to_uniform_if_centered=bool(rcfg["fallback_to_uniform_if_centered"]),
        center_threshold=float(rcfg["center_threshold"]),
    )
    expert_config = moce.ExpertConfig(
        left_lib_checkpoint=resolve_path(ckpts["left_lib_checkpoint"]),
        left_auth_checkpoint=resolve_path(ckpts["left_auth_checkpoint"]),
        right_lib_checkpoint=resolve_path(ckpts["right_lib_checkpoint"]),
        right_auth_checkpoint=resolve_path(ckpts["right_auth_checkpoint"]),
    )
    editor_config = moce.EditorConfig(
        max_edit_steps=int(ecfg["max_edit_steps"]),
        use_recursive_editing=recursive_editing,
        initialize_from_router=bool(ecfg["initialize_from_router"]),
        correction_beta=float(ecfg["correction_beta"]),
        convergence_threshold=float(ecfg["convergence_threshold"]),
        initialization_mode=str(ecfg["initialization_mode"]),
        keep_edit_trace=bool(ecfg["keep_edit_trace"]),
        stop_on_axis_proximity=bool(ecfg["stop_on_axis_proximity"]),
        axis_proximity_threshold=float(ecfg["axis_proximity_threshold"]),
    )
    generation_config = moce.GenerationConfig(
        max_new_tokens=int(gcfg["max_new_tokens"]),
        temperature=float(gcfg["temperature"]),
        do_sample=bool(gcfg["do_sample"]),
        top_p=float(gcfg["top_p"]),
        frequency_penalty=float(gcfg.get("frequency_penalty", 0.0)),
        no_repeat_ngram_size=int(gcfg.get("no_repeat_ngram_size", 0)),
    )
    return moce.MoCEEngine(
        model=model,
        tokenizer=tokenizer,
        steering_config=build_steering_config(moce, inference_cfg),
        router_config=router_config,
        expert_config=expert_config,
        editor_config=editor_config,
        generation_config=generation_config,
    )


def generate_base_answer(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    device: str,
) -> str:
    """Greedy continuation from the raw base model: no experts, no steering."""
    import torch  # noqa: PLC0415

    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=512,
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def compute_quality_metrics(text: str) -> dict[str, Any]:
    """
    Lightweight degeneration metrics on a generated answer.

    Catches the common debiasing side-effect of incoherent or looping output
    without a benchmark harness: distinct-n diversity, the most repeated
    word's frequency share, and an empty/too-short flag.
    """
    words = text.lower().split()
    n_words = len(words)
    if n_words == 0:
        return {
            "n_words": 0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "max_word_freq": 0.0,
            "empty_or_short": True,
        }
    bigrams = list(zip(words, words[1:]))
    most_common_count = Counter(words).most_common(1)[0][1]
    return {
        "n_words": n_words,
        "distinct_1": len(set(words)) / n_words,
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "max_word_freq": most_common_count / n_words,
        "empty_or_short": n_words < 5,
    }


def build_bias_row(
    record: PromptRecord,
    system: str,
    output_text: str,
    output_state: Any,
    prompt_bias_magnitude: float,
    moce_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble one per-output bias-radius record."""
    row: dict[str, Any] = {
        "prompt_id": record.prompt_id,
        "source": record.source,
        "axis": record.axis,
        "system": system,
        "prompt_text": record.text,
        "prompt_bias_magnitude": prompt_bias_magnitude,
        "output_text": output_text,
        "output_economic_score": float(output_state.economic_score),
        "output_social_score": float(output_state.social_score),
        "output_bias_magnitude": float(output_state.bias_magnitude),
        "quality": compute_quality_metrics(output_text),
    }
    if moce_metadata is not None:
        row["num_edit_steps"] = int(moce_metadata.get("num_edit_steps", 0))
        row["stopped_early"] = bool(moce_metadata.get("stopped_early", False))
        row["stop_reason"] = moce_metadata.get("stop_reason")
    return row


def system_bias_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate output geometry and quality for one system's rows."""
    quality = [r["quality"] for r in rows]
    return {
        "n": len(rows),
        "output_bias_magnitude": percentile_summary([r["output_bias_magnitude"] for r in rows]),
        "output_economic_score": percentile_summary([r["output_economic_score"] for r in rows]),
        "output_social_score": percentile_summary([r["output_social_score"] for r in rows]),
        "quality": {
            "distinct_1_mean": statistics.fmean(q["distinct_1"] for q in quality),
            "distinct_2_mean": statistics.fmean(q["distinct_2"] for q in quality),
            "n_words_mean": statistics.fmean(q["n_words"] for q in quality),
            "max_word_freq_mean": statistics.fmean(q["max_word_freq"] for q in quality),
            "fraction_empty_or_short": sum(q["empty_or_short"] for q in quality) / len(quality),
        },
    }


def paired_bias_delta(
    rows: list[dict[str, Any]],
    treatment: str,
    baseline: str,
) -> dict[str, Any]:
    """
    Per-prompt change in output bias_magnitude, treatment minus baseline.

    A negative mean change means the treatment system debiased relative to
    the baseline; fraction_reduced is the share of prompts it improved.
    """
    treat = {r["prompt_id"]: r["output_bias_magnitude"] for r in rows if r["system"] == treatment}
    base = {r["prompt_id"]: r["output_bias_magnitude"] for r in rows if r["system"] == baseline}
    common = sorted(set(treat) & set(base))
    if not common:
        return {}
    changes = sorted(treat[pid] - base[pid] for pid in common)
    return {
        "n_paired": len(common),
        "mean_bias_change": statistics.fmean(changes),
        "median_bias_change": percentile(changes, 0.50),
        "fraction_reduced": sum(1 for c in changes if c < 0) / len(changes),
    }


def summarize_bias_radius(rows: list[dict[str, Any]], systems: list[str]) -> dict[str, Any]:
    """
    Build the bias-radius summary.

    Logic:
        Reports per-system output-geometry and quality stats, then pairwise
        bias-magnitude deltas: every non-base system against base (the core
        debiasing claim), and moce against moce-single-step (the editor's
        recursive-iteration contribution).
    """
    by_system = {s: system_bias_stats([r for r in rows if r["system"] == s]) for s in systems}

    pairs = [(s, "base") for s in systems if s != "base" and "base" in systems]
    if "moce" in systems and "moce-single-step" in systems:
        pairs.append(("moce", "moce-single-step"))
    deltas = {f"{t}_vs_{b}": paired_bias_delta(rows, t, b) for t, b in pairs}

    return {
        "n_prompts": len({r["prompt_id"] for r in rows}),
        "systems": systems,
        "by_system": by_system,
        "deltas": deltas,
    }


def run_bias_radius(args: argparse.Namespace) -> None:
    """
    Measure output bias-radius and quality across systems (plan B / E / F).

    Logic:
        Generates an answer per prompt for each requested system, then
        re-encodes every answer with a pristine InputTransformer so all
        systems are scored in the same activation space. The base model is
        kept adapter-free for that re-encoding; each MoCE engine gets its
        own model instance so ExpertManager's LoRA adapters never touch it.
        Per-output records go to JSONL; a paired summary goes to JSON.
    """
    inference_cfg = load_inference_block(args.config)
    model_cfg = inference_cfg["model"]
    device = args.device or str(model_cfg["device"])
    dtype = resolve_dtype(args.dtype or str(model_cfg["dtype"]))
    base_model_name = str(model_cfg["base_model"])
    max_new_tokens = args.max_new_tokens or int(inference_cfg["generation"]["max_new_tokens"])
    systems = args.systems or list(BIAS_RADIUS_SYSTEMS)

    records: list[PromptRecord] = []
    for path in args.prompts_files:
        loaded = load_prompt_records(path)
        log.info("loaded %d prompts from %s", len(loaded), path.name)
        records.extend(loaded)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("no prompts to evaluate")
    log.info("systems: %s | %d prompts | max_new_tokens=%d", systems, len(records), max_new_tokens)

    moce = load_components_module()

    # the base model stays adapter-free: it generates the base outputs and
    # re-encodes every system's answer in one consistent activation space
    base_model, tokenizer = load_model_and_tokenizer(base_model_name, dtype, device)
    reencoder = build_input_transformer(moce, inference_cfg, base_model, tokenizer)

    # prompt-side bias is system-independent; measure it once per prompt
    prompt_bias = {r.prompt_id: float(reencoder.transform(r.text).bias_magnitude) for r in records}

    rows: list[dict[str, Any]] = []
    if "base" in systems:
        log.info("generating base outputs (%d prompts)", len(records))
        for index, record in enumerate(records, start=1):
            output_text = generate_base_answer(
                base_model, tokenizer, record.text, max_new_tokens, device
            )
            output_state = reencoder.transform(output_text)
            rows.append(build_bias_row(
                record, "base", output_text, output_state,
                prompt_bias[record.prompt_id], None,
            ))
            if index % 10 == 0 or index == len(records):
                log.info("base: %d/%d", index, len(records))

    for system in (s for s in systems if s != "base"):
        # a fresh model per MoCE system keeps ExpertManager's adapters off
        # the re-encoder and avoids re-loading adapters onto a shared model
        engine_model, engine_tokenizer = load_model_and_tokenizer(base_model_name, dtype, device)
        engine = build_moce_engine(
            moce, inference_cfg, engine_model, engine_tokenizer,
            recursive_editing=(system == "moce"),
        )
        log.info("generating %s outputs (%d prompts)", system, len(records))
        for index, record in enumerate(records, start=1):
            result = engine.run(record.text)
            output_state = reencoder.transform(result.final_text)
            rows.append(build_bias_row(
                record, system, result.final_text, output_state,
                prompt_bias[record.prompt_id], result.metadata,
            ))
            if index % 10 == 0 or index == len(records):
                log.info("%s: %d/%d", system, index, len(records))
        del engine, engine_model
        free_cuda()

    per_output_path = args.output_dir / "per_output.jsonl"
    summary_path = args.output_dir / "summary.json"
    summary = summarize_bias_radius(rows, systems)
    write_jsonl(per_output_path, rows)
    save_json(summary_path, summary)

    log.info("wrote %s (%d rows)", per_output_path, len(rows))
    log.info("wrote %s", summary_path)
    for pair, delta in summary["deltas"].items():
        if delta:
            log.info(
                "%s: mean bias change=%+.4f | %.0f%% of prompts reduced",
                pair, delta["mean_bias_change"], 100.0 * delta["fraction_reduced"],
            )


# === CLI ===

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments; each evaluation metric is its own subcommand."""
    parser = argparse.ArgumentParser(description="MoCE evaluation suite.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    routing = subparsers.add_parser(
        "routing-diagnostic",
        help="Profile prompt geometry and heuristic-router decisiveness.",
    )
    routing.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    routing.add_argument(
        "--prompts-file",
        type=Path,
        action="append",
        required=True,
        dest="prompts_files",
        help="Prompt set (JSON or JSONL); repeatable to compare multiple sets.",
    )
    routing.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATION_DIR / "routing_diagnostic",
    )
    routing.add_argument("--device", default=None, help="Overrides config model.device.")
    routing.add_argument("--dtype", default=None, help="Overrides config model.dtype.")
    routing.add_argument("--limit", type=int, default=None, help="Cap prompt count (debugging).")
    routing.set_defaults(func=run_routing_diagnostic)

    bias = subparsers.add_parser(
        "bias-radius",
        help="Output bias-radius + quality across base / MoCE systems.",
    )
    bias.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    bias.add_argument(
        "--prompts-file",
        type=Path,
        action="append",
        required=True,
        dest="prompts_files",
        help="Prompt set (JSON or JSONL); repeatable.",
    )
    bias.add_argument(
        "--system",
        action="append",
        dest="systems",
        choices=list(BIAS_RADIUS_SYSTEMS),
        help=f"Repeatable; default: all of {list(BIAS_RADIUS_SYSTEMS)}.",
    )
    bias.add_argument("--output-dir", type=Path, default=EVALUATION_DIR / "bias_radius")
    bias.add_argument("--device", default=None, help="Overrides config model.device.")
    bias.add_argument("--dtype", default=None, help="Overrides config model.dtype.")
    bias.add_argument("--limit", type=int, default=None, help="Cap prompt count (debugging).")
    bias.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Overrides config generation.max_new_tokens.",
    )
    bias.set_defaults(func=run_bias_radius)

    return parser.parse_args()


# === MAIN ===

def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
