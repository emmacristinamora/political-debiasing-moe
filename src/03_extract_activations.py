# src/03_extract_activations.py


# === IMPORTS ===

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

VALIDATED_PAIRS_DIR = PROJECT_ROOT / "data" / "steering-vectors" / "validated_pairs"
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "steering-vectors" / "activations"
REPORTS_DIR = PROJECT_ROOT / "data" / "steering-vectors" / "reports"

HARDCODED_INPUTS = {
    "economic": VALIDATED_PAIRS_DIR / "economic_pairs_validated.jsonl",
    "social": VALIDATED_PAIRS_DIR / "social_pairs_validated.jsonl",
}

HARDCODED_OUTPUTS = {
    "economic": {
        "activations_file": ACTIVATIONS_DIR / "economic_activations.pt",
        "report_file": REPORTS_DIR / "economic_activations_report.json",
    },
    "social": {
        "activations_file": ACTIVATIONS_DIR / "social_activations.pt",
        "report_file": REPORTS_DIR / "social_activations_report.json",
    },
}

VALID_AXES = {"economic", "social"}
VALID_POOLING = {"mean"}
VALID_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class ExtractionSettings:
    model_name: str
    layers: list[int]
    pooling: str
    padding: bool
    truncation: bool
    max_length: int
    device: str
    dtype: str
    batch_size: int


# === CLI ===

def parse_args() -> argparse.Namespace:
    """
    Parse runtime arguments.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract pooled hidden-state activations for validated contrastive pairs."
    )
    parser.add_argument(
        "--axis",
        type=str,
        required=True,
        choices=["economic", "social"],
        help="Axis to extract activations for.",
    )
    return parser.parse_args()


# === IO HELPERS ===

def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file.
    Args:
        path: YAML path.
    Returns:
        dict[str, Any]: Parsed config dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must parse to a dictionary: {path}")

    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file into memory.
    Args:
        path: Input JSONL path.
    Returns:
        list[dict[str, Any]]: Parsed rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Validated pairs file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Each JSONL row must be an object in {path}, line {line_number}")

            records.append(record)

    return records


def save_json(payload: dict[str, Any], path: Path) -> None:
    """
    Save a JSON file.
    Args:
        payload: JSON-serializable dictionary.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_pt(payload: dict[str, Any], path: Path) -> None:
    """
    Save a PyTorch artifact.
    Args:
        payload: Serializable payload.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# === CONFIG HELPERS ===

def build_settings(config: dict[str, Any]) -> ExtractionSettings:
    """
    Build typed extraction settings from YAML.
    Args:
        config: Full config dictionary.
    Returns:
        ExtractionSettings: Parsed extraction settings.
    """
    if "extract_activations" not in config:
        raise ValueError("Missing 'extract_activations' section in config.yaml")

    extract_cfg = config["extract_activations"]
    tokenizer_cfg = extract_cfg.get("tokenizer", {})
    runtime_cfg = extract_cfg.get("runtime", {})

    settings = ExtractionSettings(
        model_name=extract_cfg["model_name"],
        layers=list(extract_cfg["layers"]),
        pooling=extract_cfg["pooling"],
        padding=bool(tokenizer_cfg["padding"]),
        truncation=bool(tokenizer_cfg["truncation"]),
        max_length=int(tokenizer_cfg["max_length"]),
        device=str(runtime_cfg["device"]),
        dtype=str(runtime_cfg["dtype"]),
        batch_size=int(runtime_cfg["batch_size"]),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: ExtractionSettings) -> None:
    """
    Validate extraction settings early.
    Args:
        settings: Parsed settings.
    """
    if not settings.model_name:
        raise ValueError("Model name cannot be empty")

    if not settings.layers:
        raise ValueError("At least one extraction layer must be provided")

    if any(layer < 0 for layer in settings.layers):
        raise ValueError(f"Layer indices must be non-negative: {settings.layers}")

    if settings.pooling not in VALID_POOLING:
        raise ValueError(f"Unsupported pooling method: {settings.pooling}")

    if settings.dtype not in VALID_DTYPES:
        raise ValueError(f"Unsupported dtype '{settings.dtype}'. Valid options: {sorted(VALID_DTYPES)}")

    if settings.max_length <= 0:
        raise ValueError("max_length must be > 0")

    if settings.batch_size != 2:
        raise ValueError("This script expects batch_size=2 so each pair runs as [pos, neg] together.")


def get_validated_pairs_path(axis: str) -> Path:
    if axis not in HARDCODED_INPUTS:
        raise ValueError(f"Unsupported axis: {axis}")
    return HARDCODED_INPUTS[axis]


def get_hardcoded_output_paths(axis: str) -> tuple[Path, Path]:
    """
    Return hardcoded activation and report paths for an axis.
    Args:
        axis: economic or social.
    Returns:
        tuple[Path, Path]: (activations_file, report_file)
    """
    if axis not in HARDCODED_OUTPUTS:
        raise ValueError(f"Unsupported axis for hardcoded outputs: {axis}")

    outputs = HARDCODED_OUTPUTS[axis]
    return outputs["activations_file"], outputs["report_file"]


# === VALIDATION HELPERS ===

def validate_pair_records(records: list[dict[str, Any]], axis: str) -> None:
    """
    Validate that loaded records are usable for activation extraction.
    Args:
        records: Loaded validated pairs.
        axis: Axis expected for all rows.
    """
    required_fields = ["id", "axis", "statement_id", "statement", "template_id", "pos", "neg"]

    if not records:
        raise ValueError(f"No validated pair records found for axis '{axis}'")

    for index, record in enumerate(records):
        for field_name in required_fields:
            if field_name not in record:
                raise ValueError(f"Missing required field '{field_name}' in record index {index}")

        if record["axis"] != axis:
            raise ValueError(
                f"Axis mismatch in record index {index}: expected '{axis}', got '{record['axis']}'"
            )

        if not isinstance(record["pos"], str) or not record["pos"].strip():
            raise ValueError(f"Empty or invalid 'pos' in record index {index}")

        if not isinstance(record["neg"], str) or not record["neg"].strip():
            raise ValueError(f"Empty or invalid 'neg' in record index {index}")


# === MODEL HELPERS ===

def load_tokenizer_and_model(settings: ExtractionSettings) -> tuple[Any, Any]:
    """
    Load tokenizer and causal LM with hidden-state support.
    Args:
        settings: Extraction settings.
    Returns:
        tuple[Any, Any]: (tokenizer, model)
    """
    print(f"[model] Loading tokenizer: {settings.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"[model] No pad_token found — reusing eos_token ({tokenizer.eos_token!r})")
        else:
            raise ValueError(
                f"Tokenizer for {settings.model_name} has no pad_token and no eos_token to reuse."
            )

    print(f"[model] Loading model weights: {settings.model_name} | dtype={settings.dtype} | device={settings.device}")
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        dtype=VALID_DTYPES[settings.dtype],
        device_map=settings.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[model] Model loaded — {get_model_layer_count(model)} layers | hidden_dim={model.config.hidden_size}")

    return tokenizer, model


def get_model_layer_count(model: Any) -> int:
    """
    Infer the number of transformer layers from model config.
    Args:
        model: Loaded Hugging Face model.
    Returns:
        int: Number of hidden layers.
    """
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise ValueError("Model config does not expose 'num_hidden_layers'")
    return int(num_hidden_layers)


def validate_requested_layers(settings: ExtractionSettings, model: Any) -> None:
    """
    Ensure requested layer indices are valid for the loaded model.
    Args:
        settings: Extraction settings.
        model: Loaded model.
    """
    num_hidden_layers = get_model_layer_count(model)

    invalid_layers = [layer for layer in settings.layers if layer < 0 or layer >= num_hidden_layers]
    if invalid_layers:
        raise ValueError(
            f"Requested layers {invalid_layers} are out of range for model with "
            f"{num_hidden_layers} layers. Requested layers: {settings.layers}"
        )


# === POOLING ===

def mean_pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token representations using the attention mask.
    Args:
        hidden_states: Tensor of shape [batch, seq_len, hidden_dim].
        attention_mask: Tensor of shape [batch, seq_len].
    Returns:
        torch.Tensor: Pooled tensor of shape [batch, hidden_dim].
    Logic:
        Ignores padding tokens and averages only over valid tokens.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden_states with 3 dims, got shape {tuple(hidden_states.shape)}")

    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask with 2 dims, got shape {tuple(attention_mask.shape)}")

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    masked_hidden = hidden_states * mask
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    pooled = masked_hidden.sum(dim=1) / token_counts
    return pooled


# === EXTRACTION ===

def extract_pair_layer_vectors(
    pair_record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    settings: ExtractionSettings,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, int]]:
    """
    Extract pooled layer vectors for one contrastive pair.
    Args:
        pair_record: Validated pair record.
        tokenizer: Loaded tokenizer.
        model: Loaded model.
        settings: Extraction settings.
    Returns:
        tuple[dict[int, dict[str, torch.Tensor]], dict[str, int]]:
            - layer -> {"pos": tensor, "neg": tensor}
            - token counts for pos and neg
    Logic:
        Tokenize [pos, neg] together, run one forward pass, then mean-pool each selected layer.
    """
    texts = [pair_record["pos"], pair_record["neg"]]

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=settings.padding,
        truncation=settings.truncation,
        max_length=settings.max_length,
    )

    input_ids = encoded["input_ids"].to(settings.device)
    attention_mask = encoded["attention_mask"].to(settings.device)

    seq_len = input_ids.shape[1]
    if seq_len == settings.max_length:
        print(
            f"[WARNING] Pair '{pair_record['id']}' hit max_length={settings.max_length} — "
            f"one or both prompts may have been truncated. "
            f"Consider increasing max_length in config."
        )

    token_counts = {
        "pos": int(attention_mask[0].sum().item()),
        "neg": int(attention_mask[1].sum().item()),
    }

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model forward pass did not return hidden_states")

    expected_hidden_states = get_model_layer_count(model) + 1
    if len(hidden_states) != expected_hidden_states:
        raise ValueError(
            f"Unexpected number of hidden_states. Expected {expected_hidden_states}, got {len(hidden_states)}"
        )

    layer_vectors: dict[int, dict[str, torch.Tensor]] = {}

    for layer_index in settings.layers:
        hidden_state_tensor = hidden_states[layer_index + 1]
        pooled = mean_pool_hidden_states(hidden_state_tensor, attention_mask)

        pos_vector = pooled[0].detach().to(torch.float32).cpu()
        neg_vector = pooled[1].detach().to(torch.float32).cpu()

        if torch.isnan(pos_vector).any() or torch.isnan(neg_vector).any():
            raise ValueError(
                f"NaN values found in pooled activations for pair '{pair_record['id']}' at layer {layer_index}"
            )

        layer_vectors[layer_index] = {
            "pos": pos_vector,
            "neg": neg_vector,
        }

    return layer_vectors, token_counts


def build_activation_store(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    settings: ExtractionSettings,
    axis: str,
) -> dict[str, Any]:
    """
    Extract pooled activations for all validated records.
    Args:
        records: Validated pair records.
        tokenizer: Loaded tokenizer.
        model: Loaded model.
        settings: Extraction settings.
        axis: economic or social.
    Returns:
        dict[str, Any]: Serializable activation artifact for downstream vector building.
    """
    layer_storage: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: {"pos": [], "neg": []} for layer in settings.layers
    }

    pair_ids: list[str] = []
    statement_ids: list[str] = []
    template_ids: list[str] = []
    token_counts_pos: list[int] = []
    token_counts_neg: list[int] = []

    n_total = len(records)
    for i, record in enumerate(records, start=1):
        print(f"[extraction] Pair {i}/{n_total} | id={record['id']}")
        layer_vectors, token_counts = extract_pair_layer_vectors(
            pair_record=record,
            tokenizer=tokenizer,
            model=model,
            settings=settings,
        )

        pair_ids.append(record["id"])
        statement_ids.append(record["statement_id"])
        template_ids.append(record["template_id"])
        token_counts_pos.append(token_counts["pos"])
        token_counts_neg.append(token_counts["neg"])

        for layer in settings.layers:
            layer_storage[layer]["pos"].append(layer_vectors[layer]["pos"])
            layer_storage[layer]["neg"].append(layer_vectors[layer]["neg"])

    stacked_activations: dict[int, dict[str, torch.Tensor]] = {}
    for layer in settings.layers:
        pos_tensor = torch.stack(layer_storage[layer]["pos"], dim=0)
        neg_tensor = torch.stack(layer_storage[layer]["neg"], dim=0)

        if pos_tensor.ndim != 2 or neg_tensor.ndim != 2:
            raise ValueError(
                f"Expected stacked activations to be 2D at layer {layer}, "
                f"got pos {tuple(pos_tensor.shape)} and neg {tuple(neg_tensor.shape)}"
            )

        stacked_activations[layer] = {
            "pos": pos_tensor,
            "neg": neg_tensor,
        }

    artifact = {
        "meta": {
            "axis": axis,
            "model_name": settings.model_name,
            "layers": settings.layers,
            "pooling": settings.pooling,
            "max_length": settings.max_length,
            "padding": settings.padding,
            "truncation": settings.truncation,
            "dtype_forward": settings.dtype,
            "saved_dtype": "float32",
            "batch_size": settings.batch_size,
            "num_pairs": len(records),
        },
        "pair_ids": pair_ids,
        "statement_ids": statement_ids,
        "template_ids": template_ids,
        "token_counts": {
            "pos": token_counts_pos,
            "neg": token_counts_neg,
        },
        "activations": stacked_activations,
    }

    return artifact


# === REPORTING ===

def build_extraction_report(
    axis: str,
    input_path: Path,
    activations_path: Path,
    settings: ExtractionSettings,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a small extraction summary.
    Args:
        axis: economic or social.
        input_path: Validated input path.
        activations_path: Saved PT path.
        settings: Extraction settings.
        artifact: Saved activation artifact.
    Returns:
        dict[str, Any]: Summary payload.
    """
    token_counts_pos = artifact["token_counts"]["pos"]
    token_counts_neg = artifact["token_counts"]["neg"]

    avg_pos_tokens = sum(token_counts_pos) / len(token_counts_pos)
    avg_neg_tokens = sum(token_counts_neg) / len(token_counts_neg)

    hidden_dim = None
    if settings.layers:
        first_layer = settings.layers[0]
        hidden_dim = int(artifact["activations"][first_layer]["pos"].shape[1])

    return {
        "axis": axis,
        "validated_input_file": str(input_path),
        "activations_output_file": str(activations_path),
        "model_name": settings.model_name,
        "layers": settings.layers,
        "pooling": settings.pooling,
        "max_length": settings.max_length,
        "num_pairs": artifact["meta"]["num_pairs"],
        "hidden_dim": hidden_dim,
        "avg_pos_tokens": avg_pos_tokens,
        "avg_neg_tokens": avg_neg_tokens,
    }


# === MAIN ===

def main() -> None:
    """
    Run activation extraction for one axis.
    Logic:
        1. load config
        2. load validated pairs for the requested axis
        3. load tokenizer and model
        4. extract pooled activations at selected layers
        5. save one .pt artifact and one JSON report
    """
    args = parse_args()
    axis = args.axis

    if axis not in VALID_AXES:
        raise ValueError(f"Unsupported axis: {axis}")

    print(f"\n{'='*60}")
    print(f"[start] Axis: {axis}")
    print(f"[start] Config: {CONFIG_PATH}")

    config = load_yaml(CONFIG_PATH)
    settings = build_settings(config)

    print(f"[config] Model: {settings.model_name}")
    print(f"[config] Layers: {settings.layers}")
    print(f"[config] Pooling: {settings.pooling} | max_length: {settings.max_length} | dtype: {settings.dtype} | device: {settings.device}")
    print(f"{'='*60}\n")

    validated_pairs_path = get_validated_pairs_path(axis)
    activations_output_path, report_output_path = get_hardcoded_output_paths(axis)

    print(f"[io] Input:  {validated_pairs_path}")
    print(f"[io] Output: {activations_output_path}")
    print(f"[io] Report: {report_output_path}\n")

    records = load_jsonl(validated_pairs_path)
    validate_pair_records(records, axis)
    print(f"[data] Loaded {len(records)} validated pairs\n")

    tokenizer, model = load_tokenizer_and_model(settings)
    validate_requested_layers(settings, model)
    print(f"[model] Layers {settings.layers} validated against model\n")

    print(f"[extraction] Starting extraction for {len(records)} pairs across {len(settings.layers)} layers...")
    artifact = build_activation_store(
        records=records,
        tokenizer=tokenizer,
        model=model,
        settings=settings,
        axis=axis,
    )
    print(f"[extraction] Done\n")

    report = build_extraction_report(
        axis=axis,
        input_path=validated_pairs_path,
        activations_path=activations_output_path,
        settings=settings,
        artifact=artifact,
    )

    save_pt(artifact, activations_output_path)
    save_json(report, report_output_path)

    print(f"\n{'='*60}")
    print(f"[done] Axis:       {axis}")
    print(f"[done] Pairs:      {len(records)}")
    print(f"[done] Hidden dim: {report['hidden_dim']}")
    print(f"[done] Avg tokens: pos={report['avg_pos_tokens']:.1f} | neg={report['avg_neg_tokens']:.1f}")
    print(f"[done] Activations saved to: {activations_output_path}")
    print(f"[done] Report saved to:      {report_output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()