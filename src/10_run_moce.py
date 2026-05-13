# src/10_run_moce.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml


# === CONFIG ===

# src/09_moce_components.py begins with a digit and cannot be imported via
# normal "import" syntax; we load it explicitly by absolute path.
COMPONENTS_PATH = Path(__file__).resolve().parent / "09_moce_components.py"

# Top-level keys required under config.yaml's moce_inference block. Each maps
# to a downstream dataclass in 09_moce_components.py.
REQUIRED_INFERENCE_BLOCKS = (
    "model",
    "steering_vectors",
    "input_transformer",
    "expert_checkpoints",
    "router",
    "editor",
    "generation",
)

log = logging.getLogger("run_moce")


# === HELPERS: MODULE + CONFIG LOADING ===

def _load_components_module() -> Any:
    """
    Load src/09_moce_components.py via importlib.

    Returns:
        The loaded module exposing MoCE components such as Router,
        RouterConfig, InputTransformer, Editor, ExpertManager, and MoCEEngine.
    """
    spec = importlib.util.spec_from_file_location("moce_components", COMPONENTS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load components module at {COMPONENTS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["moce_components"] = module
    spec.loader.exec_module(module)
    return module


def _load_moce_inference_block(config_path: Path) -> dict[str, Any]:
    """
    Load config.yaml and return the moce_inference sub-block.

    Logic:
        Reads the YAML file, extracts the moce_inference block, and validates
        that every entry in REQUIRED_INFERENCE_BLOCKS is present. Each missing
        sub-key raises ValueError with the full key path so misconfiguration
        surfaces here rather than deep inside engine construction.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{config_path}: expected top-level mapping, got {type(raw).__name__}"
        )
    block = raw.get("moce_inference")
    if not isinstance(block, dict):
        raise ValueError(
            f"{config_path}: missing or non-mapping 'moce_inference' block"
        )
    for key in REQUIRED_INFERENCE_BLOCKS:
        if key not in block or not isinstance(block[key], dict):
            raise ValueError(
                f"{config_path}: moce_inference.{key} is missing or not a mapping"
            )
    return block


def _resolve_dtype(name: str) -> Any:
    """
    Resolve a config dtype string into a torch.dtype.

    torch is imported locally so this module's --help path stays fast.
    """
    import torch  # noqa: PLC0415

    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "half":     torch.float16,
        "float32":  torch.float32,
        "fp32":     torch.float32,
        "float":    torch.float32,
    }
    if name not in aliases:
        raise ValueError(
            f"unsupported dtype {name!r}; expected one of {sorted(aliases)}"
        )
    return aliases[name]


# === HELPERS: MODEL + ENGINE ===

def load_model_and_tokenizer(
    base_model_name: str,
    dtype: Any,
    device: str,
) -> tuple[Any, Any]:
    """
    Load the base causal LM and its tokenizer, alias pad->eos for Mistral,
    place the model on the requested device, and set eval mode.

    Mirrors src/router_training/forced_policy_runner.load_model_and_tokenizer
    to keep prefill geometry consistent between calibration data collection
    and live MoCE inference.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    log.info("loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info(
        "loading base model: %s  dtype=%s  device=%s",
        base_model_name, dtype, device,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype,
        )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def build_engine(
    inference_cfg: dict[str, Any],
    args: argparse.Namespace,
    moce: Any,
    model: Any,
    tokenizer: Any,
) -> Any:
    """
    Construct a MoCEEngine from the moce_inference block.

    Logic:
        Heuristic routing is the default; when args.calibrated is set, the
        router is constructed in calibrated mode and the checkpoint is loaded
        via Router.load_calibration_checkpoint. The editor uses the values
        from the moce_inference.editor sub-block (correction_beta=1.0 in the
        default config -- this is normal MoCE behavior, not the forced-policy
        pass-through used by the router-training runner).

    args.calibration_input_dim overrides the config's
    input_transformer.calibration_input_dim when provided; otherwise the
    config value is used. The two must agree with the calibration
    checkpoint's recorded calibration_input_dim, or the loader fails loudly.
    """
    it_cfg = inference_cfg["input_transformer"]
    sv_paths = inference_cfg["steering_vectors"]
    ckpts = inference_cfg["expert_checkpoints"]
    rcfg = inference_cfg["router"]
    ecfg = inference_cfg["editor"]
    gcfg = inference_cfg["generation"]

    steering_config = moce.SteeringVectorConfig(
        economic_vector_path=Path(sv_paths["economic_vector_path"]),
        social_vector_path=Path(sv_paths["social_vector_path"]),
        vector_method=it_cfg["vector_method"],
        use_final_aggregated_vectors=bool(it_cfg["use_final_aggregated_vectors"]),
        selected_layers=list(it_cfg["selected_layers"]),
        pooling_method=it_cfg["pooling_method"],
        use_centering=bool(it_cfg["use_centering"]),
        neutral_reference_path=it_cfg.get("neutral_reference_path"),
    )

    calibration_input_dim = (
        args.calibration_input_dim
        if args.calibration_input_dim is not None
        else int(it_cfg["calibration_input_dim"])
    )

    router_config = moce.RouterConfig(
        use_calibrated_router=bool(args.calibrated),
        beta=float(rcfg["beta"]),
        temperature=float(rcfg["temperature"]),
        calibration_input_dim=int(calibration_input_dim),
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
    )

    generation_config = moce.GenerationConfig(
        max_new_tokens=int(gcfg["max_new_tokens"]),
        temperature=float(gcfg["temperature"]),
        do_sample=bool(gcfg["do_sample"]),
        top_p=float(gcfg["top_p"]),
    )

    engine = moce.MoCEEngine(
        model=model,
        tokenizer=tokenizer,
        steering_config=steering_config,
        router_config=router_config,
        expert_config=expert_config,
        editor_config=editor_config,
        generation_config=generation_config,
    )

    if args.calibrated:
        engine.router.load_calibration_checkpoint(args.router_checkpoint)

    return engine


# === HELPERS: PROMPTS + OUTPUT ===

def iter_prompts(args: argparse.Namespace) -> Iterator[tuple[str | None, str]]:
    """
    Yield (prompt_id, prompt_text) pairs from --prompt or --prompts-file.

    Logic:
        --prompt mode emits a single (None, text) pair. --prompts-file mode
        reads one JSON object per non-empty line, requires a 'prompt_text'
        string field, and propagates an optional 'id' field as prompt_id.
        Bad rows (non-object, missing or non-string prompt_text) raise
        ValueError with line numbers so input issues surface immediately.
    """
    if args.prompt is not None:
        yield None, args.prompt
        return

    path = args.prompts_file
    if not path.is_file():
        raise FileNotFoundError(f"prompts file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        for line_index, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"{path} line {line_index}: invalid JSON ({err})"
                ) from err
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path} line {line_index}: row must be a JSON object, "
                    f"got {type(row).__name__}"
                )
            text = row.get("prompt_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{path} line {line_index}: 'prompt_text' must be a "
                    f"non-empty string"
                )
            prompt_id = row.get("id")
            if prompt_id is not None and not isinstance(prompt_id, str):
                prompt_id = str(prompt_id)
            yield prompt_id, text


def serialize_result(
    prompt_id: str | None,
    args: argparse.Namespace,
    result: Any,
) -> dict[str, Any]:
    """
    Build a JSON-safe record summarizing one engine.run output.

    Includes the router-mode tag, prior/policy/alpha mappings, the editor's
    step counters, and the prompt-state's quadrant scores. Hidden states
    and tensor artifacts are intentionally excluded.
    """
    prompt_state = result.prompt_state
    router_state = result.router_state
    editor_result = result.editor_result
    metadata = result.metadata if isinstance(result.metadata, dict) else {}

    return {
        "id": prompt_id,
        "prompt_text": result.prompt_text,
        "final_text": result.final_text,
        "router_mode": "calibrated" if args.calibrated else "heuristic",
        "bias_magnitude": float(prompt_state.bias_magnitude),
        "economic_score": float(prompt_state.economic_score),
        "social_score": float(prompt_state.social_score),
        "quadrant_scores": {k: float(v) for k, v in prompt_state.quadrant_scores.items()},
        "heuristic_prior": {k: float(v) for k, v in router_state.heuristic_prior.items()},
        "calibrated_policy": {
            k: float(v) for k, v in router_state.calibrated_policy.items()
        },
        "final_alpha": {k: float(v) for k, v in editor_result.final_alpha.items()},
        "final_alignment": {
            k: float(v) for k, v in editor_result.final_alignment.items()
        },
        "num_edit_steps": int(metadata.get("num_edit_steps", editor_result.num_steps_run)),
        "stopped_early": bool(metadata.get("stopped_early", editor_result.stopped_early)),
    }


def _format_mapping(mapping: dict[str, float]) -> str:
    """Format a canonical-quadrant mapping as 'k=0.xxx' joined by commas."""
    return ", ".join(f"{k}={v:.3f}" for k, v in mapping.items())


def format_stdout_summary(prompt_id: str | None, result: Any) -> str:
    """
    Produce a multi-line, human-readable summary of one engine.run output.
    """
    router_state = result.router_state
    editor_result = result.editor_result
    header = f"id: {prompt_id}" if prompt_id is not None else "id: -"
    lines = [
        "=" * 72,
        header,
        f"prompt:        {result.prompt_text}",
        f"final_text:    {result.final_text}",
        f"prior:         {_format_mapping(router_state.heuristic_prior)}",
        f"policy:        {_format_mapping(router_state.calibrated_policy)}",
        f"final_alpha:   {_format_mapping(editor_result.final_alpha)}",
        f"edit_steps:    {editor_result.num_steps_run}  "
        f"stopped_early={editor_result.stopped_early}",
    ]
    return "\n".join(lines)


# === CLI ===

def parse_args() -> argparse.Namespace:
    """
    Parse runtime arguments and enforce the prompt-source and calibration
    contracts.

    Logic:
        Exactly one of --prompt / --prompts-file must be supplied. Default
        router mode is heuristic; calibrated mode is opt-in and requires
        --router-checkpoint. Passing a checkpoint without --calibrated is
        rejected as misconfiguration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run MoCE inference end-to-end. Loads the base model, builds a "
            "MoCEEngine from config.yaml's moce_inference block, and decodes "
            "one or more prompts through transform -> route -> experts -> "
            "edit -> decode. Heuristic routing is the default; calibrated "
            "routing is opt-in."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config.yaml (must contain a moce_inference block).",
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt string. Mutually exclusive with --prompts-file.",
    )
    prompt_group.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help=(
            "Path to a JSONL file with one prompt per row "
            "({'prompt_text': '...', 'id': '...optional...'})."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Optional JSONL output path. When given, one row per prompt is "
            "appended with router/editor/final-text fields."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override moce_inference.model.device (e.g. cuda, cpu).",
    )

    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Enable calibrated routing. Requires --router-checkpoint.",
    )
    parser.add_argument(
        "--router-checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a calibrated-router checkpoint produced by "
            "src/router_training/. Required iff --calibrated."
        ),
    )
    parser.add_argument(
        "--calibration-input-dim",
        "--router-hidden-dim",
        dest="calibration_input_dim",
        type=int,
        default=None,
        help=(
            "Override moce_inference.input_transformer.calibration_input_dim. "
            "Used only in calibrated mode; must match the checkpoint's "
            "calibration_input_dim. --router-hidden-dim is a deprecated alias."
        ),
    )

    args = parser.parse_args()

    if args.calibrated and args.router_checkpoint is None:
        parser.error("--calibrated requires --router-checkpoint")
    if (not args.calibrated) and args.router_checkpoint is not None:
        parser.error(
            "--router-checkpoint was provided without --calibrated; "
            "remove the checkpoint or pass --calibrated"
        )
    return args


# === MAIN ===

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    moce = _load_components_module()
    inference_cfg = _load_moce_inference_block(args.config)

    device = args.device or str(inference_cfg["model"]["device"])
    dtype = _resolve_dtype(str(inference_cfg["model"]["dtype"]))
    base_model = str(inference_cfg["model"]["base_model"])

    model, tokenizer = load_model_and_tokenizer(base_model, dtype, device)
    engine = build_engine(inference_cfg, args, moce, model, tokenizer)

    mode = "calibrated" if args.calibrated else "heuristic"
    log.info("MoCE engine ready (router mode: %s)", mode)

    output_fh = None
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        output_fh = args.output_path.open("a", encoding="utf-8")

    try:
        for prompt_id, prompt_text in iter_prompts(args):
            result = engine.run(prompt_text)
            print(format_stdout_summary(prompt_id, result))
            if output_fh is not None:
                row = serialize_result(prompt_id, args, result)
                output_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_fh.flush()
    finally:
        if output_fh is not None:
            output_fh.close()


if __name__ == "__main__":
    main()
