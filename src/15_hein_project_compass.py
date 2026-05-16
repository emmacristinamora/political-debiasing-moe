# src/15_hein_project_compass.py

# Tier 2 — external ground-truth validation of the steering vectors, step 2/3.
#
# Projects each legislator's speech corpus onto the political-compass steering
# vectors. The corpus is split into fixed-length token windows; each window is
# run through Mistral-7B, its hidden states at the steering layers are
# mean-pooled and projected onto the per-layer economic and social directions,
# then the per-layer scores are combined with the same quality weights used to
# build the aggregate vectors (mirrors src/13_steering_vector_geometry). The
# per-window coordinates are kept so step 3 can estimate within-legislator
# noise; their mean is the legislator's compass position.
#
#   step 1  src/14_hein_build_dataset.py     -> legislator_dataset.jsonl
#   step 2  src/15_hein_project_compass.py   -> compass_projections.jsonl  (GPU)
#   step 3  src/16_hein_dwnominate_analysis.py -> correlation report


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = PROJECT_ROOT / "data" / "external" / "hein_dwnominate" / "legislator_dataset.jsonl"
DEFAULT_VECTORS_DIR = PROJECT_ROOT / "data" / "steering-vectors" / "vectors"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external" / "hein_dwnominate" / "compass_projections.jsonl"

DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"
AXES = ("economic", "social")
VALID_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
VALID_METHODS = ("mean_difference", "logistic_regression")


@dataclass
class AxisVectors:
    """Per-layer unit steering directions and aggregation weights for one axis."""

    layers: list[int]
    weights: dict[int, float]
    layer_vectors: dict[int, torch.Tensor]


# === HELPERS: IO ===

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tier 2 step 2 — project legislator corpora onto the compass."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vectors-dir", type=Path, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=sorted(VALID_DTYPES), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", choices=VALID_METHODS, default="mean_difference")
    parser.add_argument("--chunk-tokens", type=int, default=256,
                        help="token-window length, matching the activation-extraction length.")
    parser.add_argument("--min-chunk-tokens", type=int, default=32,
                        help="drop a trailing window shorter than this.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="token windows per forward pass.")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of legislators (debugging).")
    return parser.parse_args()


def load_legislators(path: Path) -> list[dict[str, Any]]:
    """Load the per-legislator dataset written by step 1."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


# === HELPERS: MODEL & VECTORS ===

def load_model_and_tokenizer(model_name: str, dtype: torch.dtype, device: str) -> tuple[Any, Any]:
    """Load the base Mistral model and tokenizer, placed on device in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:  # older transformers expect torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_axis_vectors(vectors_dir: Path, axis: str, method: str, device: str) -> AxisVectors:
    """Load the per-layer unit steering directions and aggregation weights for one axis."""
    path = vectors_dir / f"{axis}_vectors.pt"
    if not path.is_file():
        raise FileNotFoundError(f"vectors file not found: {path}")
    data = torch.load(path, map_location="cpu")
    aggregation = data["aggregation"][method]
    layers = [int(layer) for layer in aggregation["layers"]]
    weights = dict(zip(layers, (float(w) for w in aggregation["normalized_weights"])))
    layer_vectors: dict[int, torch.Tensor] = {}
    for layer in layers:
        vector = data["per_layer"][layer][method]["vector"].to(torch.float32)
        layer_vectors[layer] = (vector / (vector.norm() + 1e-12)).to(device)
    return AxisVectors(layers=layers, weights=weights, layer_vectors=layer_vectors)


# === HELPERS: PROJECTION ===

def chunk_token_ids(token_ids: list[int], chunk_tokens: int, min_chunk_tokens: int) -> list[list[int]]:
    """Split a token-id list into fixed-length windows, dropping a too-short tail."""
    chunks = [token_ids[start:start + chunk_tokens] for start in range(0, len(token_ids), chunk_tokens)]
    return [chunk for chunk in chunks if len(chunk) >= min_chunk_tokens]


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token representations using the attention mask (mirrors step 03)."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    return (hidden_states * mask).sum(dim=1) / token_counts


def project_batch(
    model: Any,
    chunks: list[list[int]],
    pad_token_id: int,
    axis_vectors: dict[str, AxisVectors],
    device: str,
) -> dict[str, list[float]]:
    """
    Project one batch of token windows onto every axis.

    Logic:
        Right-pads the windows, runs one forward pass with hidden states, and
        for each axis mean-pools each steering layer, projects onto that
        layer's unit direction, and combines layers with the quality weights.
    """
    width = max(len(chunk) for chunk in chunks)
    input_ids = torch.full((len(chunks), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(chunks), width), dtype=torch.long)
    for row, chunk in enumerate(chunks):
        input_ids[row, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)
        attention_mask[row, : len(chunk)] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states

    scores: dict[str, list[float]] = {}
    for axis, vectors in axis_vectors.items():
        combined = torch.zeros(len(chunks), device=device)
        for layer in vectors.layers:
            pooled = mean_pool(hidden_states[layer + 1].float(), attention_mask)
            projection = pooled @ vectors.layer_vectors[layer]
            combined = combined + vectors.weights[layer] * projection
        scores[axis] = combined.detach().cpu().tolist()
    return scores


def project_corpus(
    model: Any,
    token_ids: list[int],
    pad_token_id: int,
    axis_vectors: dict[str, AxisVectors],
    args: argparse.Namespace,
) -> dict[str, list[float]]:
    """Chunk one legislator's corpus and project every window onto both axes."""
    chunks = chunk_token_ids(token_ids, args.chunk_tokens, args.min_chunk_tokens)
    per_axis: dict[str, list[float]] = {axis: [] for axis in axis_vectors}
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start:start + args.batch_size]
        batch_scores = project_batch(model, batch, pad_token_id, axis_vectors, args.device)
        for axis, values in batch_scores.items():
            per_axis[axis].extend(values)
    return per_axis


# === MAIN ===

def mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


def stddev(values: list[float]) -> float:
    """Sample standard deviation; 0.0 when fewer than two values."""
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((v - avg) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def main() -> None:
    """Project every legislator's corpus onto the compass and write one JSONL."""
    args = parse_args()

    legislators = load_legislators(args.dataset)
    if args.limit is not None:
        legislators = legislators[: args.limit]

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, VALID_DTYPES[args.dtype], args.device
    )
    axis_vectors = {
        axis: load_axis_vectors(args.vectors_dir, axis, args.method, args.device)
        for axis in AXES
    }
    print(f"projecting {len(legislators)} legislators "
          f"({args.chunk_tokens}-token windows, batch {args.batch_size})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, legislator in enumerate(legislators, start=1):
            token_ids = tokenizer(legislator["text"], add_special_tokens=False)["input_ids"]
            per_axis = project_corpus(model, token_ids, tokenizer.pad_token_id, axis_vectors, args)
            economic, social = per_axis["economic"], per_axis["social"]
            handle.write(json.dumps({
                "icpsr": legislator["icpsr"],
                "bioname": legislator["bioname"],
                "party": legislator["party"],
                "nominate_dim1": legislator["nominate_dim1"],
                "nominate_dim2": legislator["nominate_dim2"],
                "n_chunks": len(economic),
                "economic_coord": mean(economic),
                "social_coord": mean(social),
                "economic_coord_std": stddev(economic),
                "social_coord_std": stddev(social),
                "economic_chunks": economic,
                "social_chunks": social,
            }) + "\n")
            handle.flush()
            if index % 50 == 0 or index == len(legislators):
                print(f"projected {index}/{len(legislators)}")

    print(f"wrote {args.output}")
    print(f"next: python src/16_hein_dwnominate_analysis.py")


if __name__ == "__main__":
    main()
