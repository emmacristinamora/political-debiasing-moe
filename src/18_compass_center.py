# src/18_compass_center.py

# Computes the political-compass center for a dataset of politically neutral
# prompts.  Each prompt is run through Mistral-7B-v0.1; hidden states at layer
# 20 are mean-pooled and projected onto the final (quality-weighted aggregate)
# steering vector for each axis.  The mean of all per-prompt coordinates is the
# "compass center" — the point in compass space that neutral language occupies.
#
#   input   data/neutral_prompts.jsonl          one {"text": "..."} per line
#   output  data/compass_center/center.json


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET    = PROJECT_ROOT / "data" / "neutral_prompts.jsonl"
DEFAULT_VECTORS_DIR = PROJECT_ROOT / "data" / "steering-vectors" / "vectors"
DEFAULT_OUTPUT     = PROJECT_ROOT / "data" / "compass_center" / "center.json"

DEFAULT_MODEL  = "mistralai/Mistral-7B-v0.1"
AXES           = ("economic", "social")
LAYER          = 20
VALID_DTYPES   = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
VALID_METHODS  = ("mean_difference", "logistic_regression")


# === HELPERS: IO ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project neutral prompts onto the compass and compute their center."
    )
    parser.add_argument("--dataset",     type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vectors-dir", type=Path, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--output",      type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name",  default=DEFAULT_MODEL)
    parser.add_argument("--dtype",       choices=sorted(VALID_DTYPES), default="float16")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--method",      choices=VALID_METHODS, default="mean_difference")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--max-tokens",  type=int, default=256,
                        help="truncate each prompt to this many tokens.")
    parser.add_argument("--limit",       type=int, default=None,
                        help="cap the number of prompts (debugging).")
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    """Load the neutral-prompt dataset; each row must have a 'text' field."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no rows")
    texts = [row["text"] for row in rows]
    return texts


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# === HELPERS: MODEL & VECTORS ===

def load_model_and_tokenizer(model_name: str, dtype: torch.dtype, device: str) -> tuple[Any, Any]:
    """Load the base Mistral model and tokenizer in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_final_vector(vectors_dir: Path, axis: str, method: str, device: str) -> torch.Tensor:
    """Load the final quality-weighted aggregate steering direction for one axis."""
    path = vectors_dir / f"{axis}_vectors.pt"
    if not path.is_file():
        raise FileNotFoundError(f"vectors file not found: {path}")
    data = torch.load(path, map_location="cpu")
    vector = data["final_vectors"][method].to(torch.float32)
    unit = vector / (vector.norm() + 1e-12)
    return unit.to(device)


# === HELPERS: PROJECTION ===

def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token representations using the attention mask (mirrors step 03)."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    return (hidden_states * mask).sum(dim=1) / token_counts


def project_batch(
    model: Any,
    texts: list[str],
    tokenizer: Any,
    final_vectors: dict[str, torch.Tensor],
    max_tokens: int,
    device: str,
) -> dict[str, list[float]]:
    """
    Project one batch of prompts onto both axes.

    Tokenises the texts, runs a single forward pass, extracts hidden states at
    layer 20, mean-pools over the token dimension, then dots each pooled
    representation with the final unit steering vector for each axis.
    """
    encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states[i] is the output of transformer block i-1 (index 0 = embedding).
    # Layer 20 output is at index 21.
    layer20 = outputs.hidden_states[LAYER + 1].float()
    pooled  = mean_pool(layer20, attention_mask)            # [batch, hidden_dim]

    scores: dict[str, list[float]] = {}
    for axis, unit_vec in final_vectors.items():
        scores[axis] = (pooled @ unit_vec).detach().cpu().tolist()
    return scores


# === MAIN ===

def mean_of(values: list[float]) -> float:
    return sum(values) / len(values)


def stddev_of(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean_of(values)
    return (sum((v - avg) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def main() -> None:
    """Project all neutral prompts and compute their compass center."""
    args = parse_args()

    texts = load_prompts(args.dataset)
    if args.limit is not None:
        texts = texts[: args.limit]
    print(f"loaded {len(texts)} prompts from {args.dataset}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, VALID_DTYPES[args.dtype], args.device
    )
    final_vectors = {
        axis: load_final_vector(args.vectors_dir, axis, args.method, args.device)
        for axis in AXES
    }
    print(f"projecting {len(texts)} prompts onto {AXES} (layer {LAYER}, method={args.method})")

    all_scores: dict[str, list[float]] = {axis: [] for axis in AXES}

    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        batch_scores = project_batch(
            model, batch, tokenizer, final_vectors, args.max_tokens, args.device
        )
        for axis, values in batch_scores.items():
            all_scores[axis].extend(values)
        done = min(start + args.batch_size, len(texts))
        if done % 200 == 0 or done == len(texts):
            print(f"  projected {done}/{len(texts)}")

    center = {axis: mean_of(all_scores[axis]) for axis in AXES}
    result = {
        "n_prompts": len(texts),
        "layer": LAYER,
        "method": args.method,
        "center": center,
        "per_axis": {
            axis: {
                "mean": center[axis],
                "std": stddev_of(all_scores[axis]),
                "scores": all_scores[axis],
            }
            for axis in AXES
        },
    }

    save_json(result, args.output)

    print(f"\n=== compass center ===")
    print(f"  economic : {center['economic']:+.6f}")
    print(f"  social   : {center['social']:+.6f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
