# src/05_quadrant_datasets.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

# map from --source flag to normalized JSONL filename
NORMALIZED_FILES: Dict[str, str] = {
    "allsides":            "allsides.jsonl",
    "reddit_liberal":      "reddit_liberal.jsonl",
    "reddit_conservative": "reddit_conservative.jsonl",
    "hoc":                 "uk_house_of_commons.jsonl",
    "ec_press":            "ec_press_releases.jsonl",
    "uk_press":            "uk_gov_press_releases.jsonl",
    "ire_press":           "ire_gov_press_releases.jsonl",
}

VALID_SOURCES = set(NORMALIZED_FILES.keys())


# === DATACLASSES ===

@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_tokens: int = 128

@dataclass
class ThresholdConfig:
    min_abs_econ: float = 0.10
    min_abs_soc: float  = 0.10
    min_confidence_margin: float = 0.10

@dataclass
class PoolingConfig:
    layer: int      = -1
    pooling: str    = "mean"
    max_length: int = 512
    batch_size: int = 8

@dataclass
class TopicPrototype:
    name: str
    text: str

@dataclass
class TopicConfig:
    prototypes: List[TopicPrototype] = field(default_factory=list)

@dataclass
class BuildConfig:
    source: str
    normalized_dir: Path
    output_dir: Path
    model_name_or_path: str
    econ_vectors_path: Path
    soc_vectors_path: Path
    config_yaml_path: Path
    device: str = "cuda"
    dtype: str  = "float16"
    hoc_sample_n: Optional[int] = None   # None = use all HoC speeches
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)

@dataclass
class RawDocument:
    document_id: str
    text: str
    source_name: str
    source_family: str
    language: str = "en"
    title: Optional[str] = None
    date: Optional[str] = None
    speaker_or_author: Optional[str] = None
    twitter_flag: bool = False
    raw_dataset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkRecord:
    example_id: str
    quadrant: str
    document_id: str
    chunk_id: str
    source_family: str
    source_name: str
    speaker_or_author: Optional[str]
    date: Optional[str]
    topic_primary: Optional[str]
    topic_secondary: Optional[str]
    topic_primary_score: Optional[float]
    topic_secondary_score: Optional[float]
    twitter_flag: bool
    language: str
    text: str
    n_tokens: int
    score_econ: float
    score_soc: float
    score_abs_econ: float
    score_abs_soc: float
    confidence_margin: float
    threshold_pass: bool
    selection_stage: str
    raw_dataset: Optional[str]
    title: Optional[str]
    embedding_norm: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentSummary:
    document_id: str
    source_name: str
    source_family: str
    language: str
    title: Optional[str]
    date: Optional[str]
    speaker_or_author: Optional[str]
    twitter_flag: bool
    raw_dataset: Optional[str]
    n_chunks: int
    mean_score_econ: float
    mean_score_soc: float
    std_score_econ: float
    std_score_soc: float
    dominant_quadrant: Optional[str]
    quadrant_counts: Dict[str, int]


# === IO HELPERS ===

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config at {path} must be a mapping")
    return payload


# === LOGGING ===

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# === CLI ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quadrant datasets from normalized expert corpora."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=sorted(VALID_SOURCES),
        help="Which normalized JSONL source to process.",
    )
    parser.add_argument("--config-yaml-path", type=Path, default=CONFIG_PATH)

    # Optional overrides — if omitted, values come from config.yaml
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--normalized-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--econ-vectors-path", type=Path, default=None)
    parser.add_argument("--soc-vectors-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--min-chunk-tokens", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--pooling", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--min-abs-econ", type=float, default=None)
    parser.add_argument("--min-abs-soc", type=float, default=None)
    parser.add_argument("--min-confidence-margin", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Chunks per model forward pass (default from config).")
    parser.add_argument("--hoc-sample-n", type=int, default=None,
                        help="Max HoC speeches to sample (stratified by decade×party). "
                             "Ignored for non-HoC sources. None = use all.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BuildConfig:
    yaml_payload = load_yaml(args.config_yaml_path)
    dataset_cfg   = yaml_payload.get("quadrant_dataset", {})
    chunking_cfg  = dataset_cfg.get("chunking", {})
    threshold_cfg = dataset_cfg.get("thresholds", {})
    pooling_cfg   = dataset_cfg.get("pooling", {})
    topic_cfg     = dataset_cfg.get("topics", {})

    prototypes = parse_topic_prototypes(topic_cfg)

    def _path(cli_val: Optional[Path], yaml_key: str) -> Path:
        if cli_val is not None:
            return cli_val
        raw = dataset_cfg.get(yaml_key)
        if raw is None:
            raise ValueError(f"config.yaml is missing quadrant_dataset.{yaml_key}")
        return PROJECT_ROOT / raw

    base_output_dir = _path(args.output_dir, "output_dir")
    # Each source gets its own subdirectory so runs accumulate without overwriting.
    source_output_dir = base_output_dir / args.source

    return BuildConfig(
        source=args.source,
        normalized_dir=_path(args.normalized_dir, "normalized_dir"),
        output_dir=source_output_dir,
        model_name_or_path=(
            args.model_name_or_path
            or dataset_cfg.get("model_name_or_path")
            or yaml_payload.get("extract_activations", {}).get("model_name")
        ),
        econ_vectors_path=_path(args.econ_vectors_path, "econ_vectors_path"),
        soc_vectors_path=_path(args.soc_vectors_path, "soc_vectors_path"),
        config_yaml_path=args.config_yaml_path,
        device=args.device or str(dataset_cfg.get("device", "cuda")),
        dtype=args.dtype or str(dataset_cfg.get("dtype", "float16")),
        chunking=ChunkingConfig(
            chunk_size=args.chunk_size or int(chunking_cfg.get("chunk_size", 512)),
            chunk_overlap=args.chunk_overlap or int(chunking_cfg.get("chunk_overlap", 128)),
            min_chunk_tokens=args.min_chunk_tokens or int(chunking_cfg.get("min_chunk_tokens", 128)),
        ),
        thresholds=ThresholdConfig(
            min_abs_econ=args.min_abs_econ if args.min_abs_econ is not None else float(threshold_cfg.get("min_abs_econ", 0.10)),
            min_abs_soc=args.min_abs_soc if args.min_abs_soc is not None else float(threshold_cfg.get("min_abs_soc", 0.10)),
            min_confidence_margin=args.min_confidence_margin if args.min_confidence_margin is not None else float(threshold_cfg.get("min_confidence_margin", 0.10)),
        ),
        pooling=PoolingConfig(
            layer=args.layer if args.layer is not None else int(pooling_cfg.get("layer", -1)),
            pooling=args.pooling or str(pooling_cfg.get("pooling", "mean")),
            max_length=args.max_length or int(pooling_cfg.get("max_length", 512)),
            batch_size=args.batch_size or int(pooling_cfg.get("batch_size", 8)),
        ),
        topics=TopicConfig(prototypes=prototypes),
        hoc_sample_n=(
            args.hoc_sample_n if args.hoc_sample_n is not None
            else dataset_cfg.get("hoc_sample_n")  # None means use all
        ),
    )


def parse_topic_prototypes(topic_cfg: Dict[str, Any]) -> List[TopicPrototype]:
    raw_prototypes = topic_cfg.get("prototypes")
    if raw_prototypes is None:
        raise ValueError("config.yaml must contain quadrant_dataset.topics.prototypes")
    if not isinstance(raw_prototypes, list) or not raw_prototypes:
        raise ValueError("quadrant_dataset.topics.prototypes must be a non-empty list")

    prototypes: List[TopicPrototype] = []
    for item in raw_prototypes:
        if not isinstance(item, dict):
            raise ValueError("Each topic prototype must be a mapping with 'name' and 'text'")
        name = str(item.get("name", "")).strip()
        text = str(item.get("text", "")).strip()
        if not name or not text:
            raise ValueError("Each topic prototype requires non-empty 'name' and 'text'")
        prototypes.append(TopicPrototype(name=name, text=text))
    return prototypes


# === RAW DATA INGESTION ===

def _build_hoc_sample_index(
    jsonl_path: Path,
    n_sample: int,
    seed: int = 42,
) -> Optional[Set[int]]:
    """
    First pass over the HoC JSONL: build strata by (decade, party), then
    sample proportionally so every stratum is represented.

    Returns a set of line numbers to keep, or None if n_sample >= total
    (meaning the caller should use all lines).
    """
    rng = random.Random(seed)
    strata: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    logging.info("[hoc_sample] first pass: building strata from %s", jsonl_path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            date   = row.get("date") or ""
            decade = (date[:3] + "0") if len(date) >= 4 else "unknown"
            party  = (row.get("metadata") or {}).get("party") or "unknown"
            strata[(decade, party)].append(line_num)

    total = sum(len(v) for v in strata.values())
    logging.info("[hoc_sample] %d speeches across %d strata", total, len(strata))

    if n_sample >= total:
        logging.info("[hoc_sample] n_sample=%d >= total=%d, keeping all", n_sample, total)
        return None

    selected: Set[int] = set()
    for indices in strata.values():
        stratum_n = max(1, round(n_sample * len(indices) / total))
        selected.update(rng.sample(indices, min(stratum_n, len(indices))))

    logging.info(
        "[hoc_sample] sampled %d speeches (%.1f%% of %d total)",
        len(selected), 100.0 * len(selected) / total, total,
    )
    return selected


def iter_raw_documents(config: BuildConfig) -> Iterator[RawDocument]:
    """
    Stream RawDocument objects from one normalized JSONL file.

    For the HoC source, applies stratified subsampling when hoc_sample_n is
    set (two-pass: first pass builds strata, second pass yields kept lines).
    All other sources stream in a single pass.

    The normalized schema places all canonical fields at the top level and
    source-specific extras inside the nested 'metadata' dict.
    """
    jsonl_filename = NORMALIZED_FILES[config.source]
    jsonl_path = config.normalized_dir / jsonl_filename

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Normalized JSONL not found: {jsonl_path}\n"
            f"Run normalize_corpora.py --source {config.source} first."
        )

    # HoC subsampling: build the keep-set in a first pass.
    sample_index: Optional[Set[int]] = None
    if config.source == "hoc" and config.hoc_sample_n is not None:
        sample_index = _build_hoc_sample_index(jsonl_path, config.hoc_sample_n)

    logging.info("[ingest] reading %s", jsonl_path)
    n_read = n_skipped = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if sample_index is not None and line_num not in sample_index:
                continue

            row = json.loads(line)
            n_read += 1

            raw_text = row.get("text", "")
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
            text = normalize_text(raw_text)
            if not text:
                n_skipped += 1
                continue

            yield RawDocument(
                document_id=str(row["document_id"]),
                text=text,
                source_name=str(row["source_name"]),
                source_family=str(row["source_family"]),
                language=str(row.get("language") or "en"),
                title=row.get("title"),
                date=row.get("date"),
                speaker_or_author=row.get("speaker_or_author"),
                twitter_flag=bool(row.get("twitter_flag", False)),
                raw_dataset=row.get("raw_dataset"),
                metadata=row.get("metadata") or {},
            )

    logging.info(
        "[ingest] %s: read=%d skipped=%d yielded=%d",
        config.source, n_read, n_skipped, n_read - n_skipped,
    )


# === TEXT HELPERS ===

def normalize_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = text.replace("​", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()

def count_tokens(text: str) -> int:
    return len(text.split())


# === MODEL HELPERS ===

def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    lowered = dtype_name.lower()
    if lowered in {"float16", "fp16", "half"}:
        return torch.float16
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if lowered in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(config: BuildConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    torch_dtype = resolve_torch_dtype(config.dtype)
    logging.info("[model] loading tokenizer: %s", config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("[model] loading model: dtype=%s device=%s", config.dtype, config.device)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
    )
    model.eval()
    model.to(config.device)
    return model, tokenizer


def load_steering_vectors(config: BuildConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the final logistic-regression steering vectors from the two separate
    .pt files produced by script 04.

    Each file has the structure:
        payload["final_vectors"]["logistic_regression"] → (hidden_dim,) float32

    Sign conventions (enforced in script 04):
        econ  positive = econ_right,   negative = econ_left
        soc   positive = authoritarian, negative = libertarian
    """
    econ_vector = _load_final_vector(config.econ_vectors_path, axis="economic")
    soc_vector  = _load_final_vector(config.soc_vectors_path,  axis="social")
    return econ_vector, soc_vector


def _load_final_vector(path: Path, axis: str) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Steering vector file not found ({axis}): {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}, got {type(payload)}")
    try:
        vector = payload["final_vectors"]["logistic_regression"]
    except KeyError as exc:
        raise ValueError(
            f"Expected payload['final_vectors']['logistic_regression'] in {path}"
        ) from exc
    vector = torch.as_tensor(vector, dtype=torch.float32)
    if vector.ndim != 1:
        raise ValueError(f"Expected 1-D steering vector in {path}, got shape {tuple(vector.shape)}")
    return normalize_vector(vector)


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(vector, p=2)
    if norm.item() == 0.0:
        raise ValueError("Cannot normalize a zero-norm vector")
    return vector / norm


def encode_texts_batch(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: BuildConfig,
) -> List[torch.Tensor]:
    """
    Run a batch of texts through the model in one forward pass.
    Returns one normalized pooled hidden-state vector per text.

    Layer indexing matches script 03: hidden_states[layer + 1] for layer >= 0.
    layer=-1 uses the final hidden state via PyTorch negative indexing.
    Padding is added to equalize sequence lengths within the batch.
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=config.pooling.max_length,
        padding=True,
    )
    encoded = {k: v.to(config.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model did not return hidden_states. Ensure output_hidden_states=True.")

    layer_idx = config.pooling.layer
    if layer_idx >= 0:
        hidden_batch = hidden_states[layer_idx + 1]   # [B, seq_len, hidden_dim]
    else:
        hidden_batch = hidden_states[layer_idx]

    attention_mask = encoded["attention_mask"]        # [B, seq_len]

    embeddings: List[torch.Tensor] = []
    for i in range(len(texts)):
        pooled = pool_hidden_states(hidden_batch[i], attention_mask[i], config.pooling.pooling)
        embeddings.append(normalize_vector(pooled.detach().float().cpu()))

    return embeddings


def encode_text(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: BuildConfig,
) -> torch.Tensor:
    """Single-text convenience wrapper used for prototype embedding at startup."""
    return encode_texts_batch([text], model, tokenizer, config)[0]


def pool_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        masked = hidden * mask
        denom = mask.sum(dim=0).clamp(min=1.0)
        return masked.sum(dim=0) / denom
    if pooling == "last_token":
        last_idx = max(int(attention_mask.sum().item()) - 1, 0)
        return hidden[last_idx]
    raise ValueError(f"Unsupported pooling: {pooling}")


# === CHUNKING ===

def chunk_document(
    document: RawDocument,
    config: BuildConfig,
) -> List[Tuple[str, str, int]]:
    """
    Split document text into overlapping token-level windows.
    Returns list of (chunk_id, chunk_text, token_count).
    """
    tokens = document.text.split()
    if not tokens:
        return []

    size    = config.chunking.chunk_size
    overlap = config.chunking.chunk_overlap
    step    = size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: List[Tuple[str, str, int]] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + size, len(tokens))
        window = tokens[start:end]
        if len(window) >= config.chunking.min_chunk_tokens:
            chunk_id = f"{document.document_id}_chunk{chunk_index:04d}"
            chunks.append((chunk_id, " ".join(window), len(window)))
            chunk_index += 1
        if end == len(tokens):
            break
        start += step

    return chunks


# === PROJECTION AND TOPIC LABELING ===

def compute_chunk_scores(
    chunk_embedding: torch.Tensor,
    econ_vector: torch.Tensor,
    soc_vector: torch.Tensor,
) -> Tuple[float, float]:
    score_econ = float(torch.dot(chunk_embedding, econ_vector).item())
    score_soc  = float(torch.dot(chunk_embedding, soc_vector).item())
    return score_econ, score_soc


def quadrant_from_scores(score_econ: float, score_soc: float) -> str:
    """
    Sign convention (from script 04):
        econ  positive = econ_right,    negative = econ_left
        soc   positive = authoritarian, negative = libertarian

    Quadrant mapping:
        right_auth = econ_right + authoritarian (econ >= 0, soc >= 0)
        left_auth  = econ_left  + authoritarian (econ < 0,  soc >= 0)
        left_lib   = econ_left  + libertarian   (econ < 0,  soc < 0)
        right_lib  = econ_right + libertarian   (econ >= 0, soc < 0)
    """
    if score_econ >= 0 and score_soc >= 0:
        return "right_auth"
    if score_econ < 0 and score_soc >= 0:
        return "left_auth"
    if score_econ < 0 and score_soc < 0:
        return "left_lib"
    return "right_lib"


def compute_confidence_margin(score_econ: float, score_soc: float) -> float:
    return min(abs(score_econ), abs(score_soc))


def passes_thresholds(score_econ: float, score_soc: float, config: BuildConfig) -> bool:
    return (
        abs(score_econ) >= config.thresholds.min_abs_econ
        and abs(score_soc) >= config.thresholds.min_abs_soc
        and compute_confidence_margin(score_econ, score_soc) >= config.thresholds.min_confidence_margin
    )


def build_topic_prototype_embeddings(
    config: BuildConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, torch.Tensor]:
    logging.info("[topics] embedding %d prototypes", len(config.topics.prototypes))
    embeddings: Dict[str, torch.Tensor] = {}
    for prototype in config.topics.prototypes:
        embeddings[prototype.name] = encode_text(prototype.text, model, tokenizer, config)
    return embeddings


def assign_topic_label(
    chunk_embedding: torch.Tensor,
    prototype_embeddings: Dict[str, torch.Tensor],
) -> Tuple[str, Optional[str], float, Optional[float]]:
    if not prototype_embeddings:
        return "other", None, 0.0, None

    scored = [
        (name, float(F.cosine_similarity(
            chunk_embedding.unsqueeze(0),
            emb.unsqueeze(0),
            dim=1,
        ).item()))
        for name, emb in prototype_embeddings.items()
    ]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    primary_name, primary_score = ranked[0]
    secondary_name: Optional[str] = None
    secondary_score: Optional[float] = None
    if len(ranked) > 1:
        secondary_name, secondary_score = ranked[1]
    return primary_name, secondary_name, primary_score, secondary_score


# === DATASET BUILDING ===

def build_scored_chunks(
    config: BuildConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    econ_vector: torch.Tensor,
    soc_vector: torch.Tensor,
    prototype_embeddings: Dict[str, torch.Tensor],
) -> Tuple[List[ChunkRecord], List[DocumentSummary]]:
    scored_chunks: List[ChunkRecord] = []
    # doc_id -> (document, list of its ChunkRecords) for summary building
    doc_chunks: Dict[str, Tuple[RawDocument, List[ChunkRecord]]] = {}

    n_docs = 0
    n_chunks_total = 0
    n_retained = 0

    # Each item: (document, chunk_id, chunk_text, n_tokens)
    batch: List[Tuple[RawDocument, str, str, int]] = []

    def flush(items: List[Tuple[RawDocument, str, str, int]]) -> None:
        nonlocal n_chunks_total, n_retained
        if not items:
            return

        embeddings = encode_texts_batch(
            [item[2] for item in items], model, tokenizer, config
        )

        for (document, chunk_id, chunk_text, n_tokens), embedding in zip(items, embeddings):
            n_chunks_total += 1
            score_econ, score_soc = compute_chunk_scores(embedding, econ_vector, soc_vector)
            quadrant = quadrant_from_scores(score_econ, score_soc)
            passed   = passes_thresholds(score_econ, score_soc, config)

            topic_primary: Optional[str]         = None
            topic_secondary: Optional[str]       = None
            topic_primary_score: Optional[float] = None
            topic_secondary_score: Optional[float] = None
            selection_stage = "scored"

            if passed:
                n_retained += 1
                (
                    topic_primary,
                    topic_secondary,
                    topic_primary_score,
                    topic_secondary_score,
                ) = assign_topic_label(embedding, prototype_embeddings)
                selection_stage = "retained"

            record = ChunkRecord(
                example_id=f"{quadrant}_{chunk_id}",
                quadrant=quadrant,
                document_id=document.document_id,
                chunk_id=chunk_id,
                source_family=document.source_family,
                source_name=document.source_name,
                speaker_or_author=document.speaker_or_author,
                date=document.date,
                topic_primary=topic_primary,
                topic_secondary=topic_secondary,
                topic_primary_score=topic_primary_score,
                topic_secondary_score=topic_secondary_score,
                twitter_flag=document.twitter_flag,
                language=document.language,
                text=chunk_text,
                n_tokens=n_tokens,
                score_econ=score_econ,
                score_soc=score_soc,
                score_abs_econ=abs(score_econ),
                score_abs_soc=abs(score_soc),
                confidence_margin=compute_confidence_margin(score_econ, score_soc),
                threshold_pass=passed,
                selection_stage=selection_stage,
                raw_dataset=document.raw_dataset,
                title=document.title,
                embedding_norm=float(torch.norm(embedding, p=2).item()),
                metadata=document.metadata.copy(),
            )
            scored_chunks.append(record)

            if document.document_id not in doc_chunks:
                doc_chunks[document.document_id] = (document, [])
            doc_chunks[document.document_id][1].append(record)

    for document in iter_raw_documents(config):
        n_docs += 1
        for chunk_id, chunk_text, n_tokens in chunk_document(document, config):
            batch.append((document, chunk_id, chunk_text, n_tokens))
            if len(batch) >= config.pooling.batch_size:
                flush(batch)
                batch = []

        if n_docs % 500 == 0:
            logging.info(
                "[progress] docs=%d chunks=%d retained=%d",
                n_docs, n_chunks_total, n_retained,
            )

    flush(batch)  # remaining partial batch

    document_summaries = [
        summarize_document(doc, chunk_rows)
        for doc, chunk_rows in doc_chunks.values()
        if chunk_rows
    ]

    logging.info(
        "[done] docs=%d chunks=%d retained=%d (%.1f%%)",
        n_docs, n_chunks_total, n_retained,
        100.0 * n_retained / max(n_chunks_total, 1),
    )
    return scored_chunks, document_summaries


def summarize_document(document: RawDocument, chunk_rows: List[ChunkRecord]) -> DocumentSummary:
    econ_scores = [r.score_econ for r in chunk_rows]
    soc_scores  = [r.score_soc  for r in chunk_rows]
    quadrant_counts = dict(Counter(r.quadrant for r in chunk_rows))
    dominant_quadrant = max(quadrant_counts.items(), key=lambda x: x[1])[0] if quadrant_counts else None

    return DocumentSummary(
        document_id=document.document_id,
        source_name=document.source_name,
        source_family=document.source_family,
        language=document.language,
        title=document.title,
        date=document.date,
        speaker_or_author=document.speaker_or_author,
        twitter_flag=document.twitter_flag,
        raw_dataset=document.raw_dataset,
        n_chunks=len(chunk_rows),
        mean_score_econ=_mean(econ_scores),
        mean_score_soc=_mean(soc_scores),
        std_score_econ=_std(econ_scores),
        std_score_soc=_std(soc_scores),
        dominant_quadrant=dominant_quadrant,
        quadrant_counts=quadrant_counts,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


# === FILTERING ===

def retained_by_quadrant(scored_chunks: List[ChunkRecord]) -> Dict[str, List[ChunkRecord]]:
    grouped: Dict[str, List[ChunkRecord]] = defaultdict(list)
    for row in scored_chunks:
        if row.threshold_pass:
            grouped[row.quadrant].append(row)
    return grouped


# === REPORTING ===

def summarize_rows(rows: List[ChunkRecord]) -> Dict[str, Any]:
    return {
        "n_rows":                  len(rows),
        "n_documents":             len({r.document_id for r in rows}),
        "n_twitter":               sum(1 for r in rows if r.twitter_flag),
        "source_family_counts":    dict(Counter(r.source_family for r in rows)),
        "source_name_counts":      dict(Counter(r.source_name for r in rows)),
        "topic_counts":            dict(Counter((r.topic_primary or "unlabeled") for r in rows)),
        "mean_abs_econ":           _mean([r.score_abs_econ for r in rows]),
        "mean_abs_soc":            _mean([r.score_abs_soc for r in rows]),
        "mean_confidence_margin":  _mean([r.confidence_margin for r in rows]),
        "mean_tokens":             _mean([float(r.n_tokens) for r in rows]),
    }


def build_quadrant_report(rows: List[ChunkRecord], quadrant: str) -> Dict[str, Any]:
    source_rows: Dict[str, List[ChunkRecord]] = defaultdict(list)
    for row in rows:
        source_rows[row.source_family].append(row)

    source_topic_counts: Dict[str, Dict[str, int]] = {}
    source_topic_proportions: Dict[str, Dict[str, float]] = {}
    for family, family_rows in source_rows.items():
        counts = Counter((r.topic_primary or "unlabeled") for r in family_rows)
        source_topic_counts[family] = dict(counts)
        source_topic_proportions[family] = _proportion_dict(counts)

    return {
        "quadrant":                 quadrant,
        "summary":                  summarize_rows(rows),
        "source_proportions":       _proportion_dict(Counter(r.source_family for r in rows)),
        "source_topic_counts":      source_topic_counts,
        "source_topic_proportions": source_topic_proportions,
        "top_examples":             _build_top_examples(rows),
    }


def _proportion_dict(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _build_top_examples(rows: List[ChunkRecord], top_k: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: r.confidence_margin, reverse=True)[:top_k]
    return [
        {
            "example_id":       r.example_id,
            "document_id":      r.document_id,
            "topic_primary":    r.topic_primary,
            "source_family":    r.source_family,
            "source_name":      r.source_name,
            "score_econ":       r.score_econ,
            "score_soc":        r.score_soc,
            "confidence_margin":r.confidence_margin,
            "text_preview":     r.text[:300],
        }
        for r in ranked
    ]


# === SERIALIZATION ===

def chunk_record_to_dict(row: ChunkRecord) -> Dict[str, Any]:
    return asdict(row)

def document_summary_to_dict(row: DocumentSummary) -> Dict[str, Any]:
    return asdict(row)

def build_config_snapshot(config: BuildConfig) -> Dict[str, Any]:
    return {
        "source":              config.source,
        "normalized_dir":      str(config.normalized_dir),
        "output_dir":          str(config.output_dir),
        "model_name_or_path":  config.model_name_or_path,
        "econ_vectors_path":   str(config.econ_vectors_path),
        "soc_vectors_path":    str(config.soc_vectors_path),
        "device":              config.device,
        "dtype":               config.dtype,
        "chunking":            asdict(config.chunking),
        "thresholds":          asdict(config.thresholds),
        "pooling":             asdict(config.pooling),
        "n_topic_prototypes":  len(config.topics.prototypes),
        "topic_names":         [p.name for p in config.topics.prototypes],
        "batch_size":          config.pooling.batch_size,
        "hoc_sample_n":        config.hoc_sample_n,
    }


# === MAIN ===

def main() -> None:
    setup_logging()
    args   = parse_args()
    config = build_config(args)

    logging.info("source=%s output=%s", config.source, config.output_dir)
    ensure_dir(config.output_dir)
    save_json(config.output_dir / "build_config_snapshot.json", build_config_snapshot(config))

    logging.info("[setup] loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(config)

    logging.info("[setup] loading steering vectors")
    econ_vector, soc_vector = load_steering_vectors(config)
    logging.info("[setup] econ_vector shape=%s soc_vector shape=%s", tuple(econ_vector.shape), tuple(soc_vector.shape))

    logging.info("[setup] building topic prototype embeddings")
    prototype_embeddings = build_topic_prototype_embeddings(config, model, tokenizer)

    logging.info("[run] scoring chunks")
    scored_chunks, document_summaries = build_scored_chunks(
        config=config,
        model=model,
        tokenizer=tokenizer,
        econ_vector=econ_vector,
        soc_vector=soc_vector,
        prototype_embeddings=prototype_embeddings,
    )

    logging.info("[save] writing scored_chunks and document_summaries")
    save_jsonl(config.output_dir / "scored_chunks.jsonl",     (chunk_record_to_dict(r)    for r in scored_chunks))
    save_jsonl(config.output_dir / "document_summaries.jsonl",(document_summary_to_dict(r) for r in document_summaries))

    retained_groups = retained_by_quadrant(scored_chunks)

    for quadrant in ["right_auth", "left_auth", "left_lib", "right_lib"]:
        quadrant_dir  = config.output_dir / quadrant
        retained_rows = retained_groups.get(quadrant, [])
        ensure_dir(quadrant_dir)
        save_jsonl(quadrant_dir / "retained.jsonl", (chunk_record_to_dict(r) for r in retained_rows))
        save_json(quadrant_dir  / "report.json",    build_quadrant_report(retained_rows, quadrant))
        logging.info("%s | retained=%d", quadrant, len(retained_rows))

    logging.info("[done] source=%s total_chunks=%d", config.source, len(scored_chunks))


if __name__ == "__main__":
    main()
