#!/usr/bin/env python3
"""
Dry-run test for src/05_quadrant_datasets.py.

What is tested with REAL data:
    - config.yaml parsing and BuildConfig construction
    - steering vector loading from the actual .pt files
    - iter_raw_documents (reads real normalized JSONL)
    - HoC stratified subsampling logic (first-pass index building)
    - chunk_document (pure Python sliding window)
    - quadrant assignment, threshold checks, topic labeling
    - report building and output file writing

What is MOCKED:
    - load_model_and_tokenizer  → returns (None, None)
    - encode_texts_batch        → returns random unit vectors (fixed seed)
      (this also covers encode_text and build_topic_prototype_embeddings)

Why thresholds are set to 0.0:
    Random unit vectors in 4096-d have expected |dot product| ≈ 0.016,
    so almost nothing would pass the default 0.10 threshold.
    Setting all thresholds to 0.0 ensures every chunk is retained,
    which lets us verify the full downstream path (topic labeling, reports).

Usage (from project root, lt-proj env):
    python batch/dry_run_05.py

Output lands in /tmp/quadrant_dry_run/  (not the real data directory).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# ── load the module ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH  = PROJECT_ROOT / "src" / "05_quadrant_datasets.py"

print(f"[dry-run] loading module from {MODULE_PATH}")
spec = importlib.util.spec_from_file_location("s05", MODULE_PATH)
m    = importlib.util.module_from_spec(spec)
sys.modules["s05"] = m   # required so @dataclass can resolve its module namespace
spec.loader.exec_module(m)
print("[dry-run] module loaded ok")

# ── mocks ────────────────────────────────────────────────────────────────────

HIDDEN_DIM = 4096   # Mistral-7B hidden dimension

def _mock_encode_texts_batch(texts, model, tokenizer, config):
    """Return deterministic random unit vectors — no GPU needed."""
    torch.manual_seed(42)
    vecs  = torch.randn(len(texts), HIDDEN_DIM)
    norms = vecs.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    return list(vecs / norms)

def _mock_load_model_and_tokenizer(config):
    return MagicMock(), MagicMock()

# ── helper: run one source through the full pipeline ────────────────────────

def run_dry(source: str, max_docs: int, extra_cli: list[str] | None = None) -> bool:
    """
    Run the full pipeline for one source with mocked model.
    Returns True on success, False on failure.
    """
    out_dir = f"/tmp/quadrant_dry_run/{source}"

    sys.argv = [
        "05_quadrant_datasets.py",
        "--source",               source,
        "--config-yaml-path",     str(PROJECT_ROOT / "config" / "config.yaml"),
        "--output-dir",           "/tmp/quadrant_dry_run",
        "--min-abs-econ",         "0.0",
        "--min-abs-soc",          "0.0",
        "--min-confidence-margin","0.0",
    ]
    if extra_cli:
        sys.argv.extend(extra_cli)

    # Patch iter_raw_documents to cap at max_docs
    original_iter = m.iter_raw_documents

    def _capped_iter(config):
        for i, doc in enumerate(original_iter(config)):
            if i >= max_docs:
                print(f"[dry-run] reached max_docs={max_docs}, stopping ingestion")
                break
            yield doc

    print(f"\n{'='*60}")
    print(f"[dry-run] source={source}  max_docs={max_docs}  out={out_dir}")
    print(f"{'='*60}")

    try:
        with (
            patch.object(m, "load_model_and_tokenizer", _mock_load_model_and_tokenizer),
            patch.object(m, "encode_texts_batch",       _mock_encode_texts_batch),
            patch.object(m, "iter_raw_documents",       _capped_iter),
        ):
            args   = m.parse_args()
            config = m.build_config(args)
            m.ensure_dir(config.output_dir)
            m.save_json(
                config.output_dir / "build_config_snapshot.json",
                m.build_config_snapshot(config),
            )

            print(f"[dry-run] config ok")
            print(f"  normalized_dir  : {config.normalized_dir}")
            print(f"  output_dir      : {config.output_dir}")
            print(f"  model           : {config.model_name_or_path}")
            print(f"  batch_size      : {config.pooling.batch_size}")
            print(f"  hoc_sample_n    : {config.hoc_sample_n}")
            print(f"  thresholds      : econ={config.thresholds.min_abs_econ} "
                  f"soc={config.thresholds.min_abs_soc} "
                  f"margin={config.thresholds.min_confidence_margin}")
            print(f"  n_prototypes    : {len(config.topics.prototypes)}")

            # Real steering vector load
            print("\n[dry-run] loading real steering vectors...")
            econ_vec, soc_vec = m.load_steering_vectors(config)
            print(f"  econ_vec shape={tuple(econ_vec.shape)} norm={econ_vec.norm().item():.4f}")
            print(f"  soc_vec  shape={tuple(soc_vec.shape)} norm={soc_vec.norm().item():.4f}")

            # Prototype embeddings (mocked model → random vectors)
            print("\n[dry-run] building topic prototype embeddings (mocked)...")
            model_stub, tok_stub = _mock_load_model_and_tokenizer(config)
            prototype_embeddings = m.build_topic_prototype_embeddings(
                config, model_stub, tok_stub
            )
            print(f"  embedded {len(prototype_embeddings)} prototypes: "
                  f"{list(prototype_embeddings.keys())}")

            # Full pipeline
            print("\n[dry-run] running build_scored_chunks (mocked encode_texts_batch)...")
            scored_chunks, doc_summaries = m.build_scored_chunks(
                config=config,
                model=model_stub,
                tokenizer=tok_stub,
                econ_vector=econ_vec,
                soc_vector=soc_vec,
                prototype_embeddings=prototype_embeddings,
            )

            print(f"\n[dry-run] pipeline done")
            print(f"  scored_chunks   : {len(scored_chunks)}")
            print(f"  doc_summaries   : {len(doc_summaries)}")

            # Write outputs
            m.save_jsonl(
                Path(out_dir) / "scored_chunks.jsonl",
                (m.chunk_record_to_dict(r) for r in scored_chunks),
            )
            m.save_jsonl(
                Path(out_dir) / "document_summaries.jsonl",
                (m.document_summary_to_dict(r) for r in doc_summaries),
            )

            retained_groups = m.retained_by_quadrant(scored_chunks)
            for quadrant in ["q1", "q2", "q3", "q4"]:
                qdir  = Path(out_dir) / quadrant
                rows  = retained_groups.get(quadrant, [])
                m.ensure_dir(qdir)
                m.save_jsonl(qdir / "retained.jsonl",
                             (m.chunk_record_to_dict(r) for r in rows))
                m.save_json(qdir  / "report.json",
                            m.build_quadrant_report(rows, quadrant))
                print(f"  {quadrant}: {len(rows)} retained chunks")

        _verify_outputs(out_dir, scored_chunks)
        print(f"\n[dry-run] PASS  source={source}")
        return True

    except Exception:
        print(f"\n[dry-run] FAIL  source={source}")
        traceback.print_exc()
        return False


def _verify_outputs(out_dir: str, scored_chunks: list) -> None:
    """Spot-check output file schemas."""
    print("\n[dry-run] verifying output schemas...")

    # scored_chunks.jsonl — check first row has all expected keys
    sc_path = Path(out_dir) / "scored_chunks.jsonl"
    assert sc_path.exists(), f"scored_chunks.jsonl not found at {sc_path}"
    with sc_path.open() as f:
        first = json.loads(f.readline())
    required_keys = {
        "example_id", "quadrant", "document_id", "chunk_id",
        "source_family", "source_name", "text", "n_tokens",
        "score_econ", "score_soc", "score_abs_econ", "score_abs_soc",
        "confidence_margin", "threshold_pass", "selection_stage",
        "topic_primary", "topic_secondary", "twitter_flag", "language",
        "embedding_norm", "metadata",
    }
    missing = required_keys - set(first.keys())
    assert not missing, f"scored_chunks row missing keys: {missing}"
    print(f"  scored_chunks schema ok (first row example_id={first['example_id']!r})")

    # quadrant reports — check structure
    for q in ["q1", "q2", "q3", "q4"]:
        report_path = Path(out_dir) / q / "report.json"
        assert report_path.exists(), f"{report_path} not found"
        with report_path.open() as f:
            report = json.load(f)
        assert "quadrant"   in report, f"{q}/report.json missing 'quadrant'"
        assert "summary"    in report, f"{q}/report.json missing 'summary'"
        assert "top_examples" in report, f"{q}/report.json missing 'top_examples'"
        print(f"  {q}/report.json ok  (n_rows={report['summary']['n_rows']})")

    if scored_chunks:
        r = scored_chunks[0]
        assert isinstance(r.score_econ, float), "score_econ is not float"
        assert isinstance(r.score_soc,  float), "score_soc is not float"
        assert r.quadrant in {"q1", "q2", "q3", "q4"}, f"bad quadrant: {r.quadrant}"
        assert r.selection_stage in {"scored", "retained"}, \
            f"bad selection_stage: {r.selection_stage}"
    print("  ChunkRecord fields ok")


# ── standalone HoC sampling test ─────────────────────────────────────────────

def test_hoc_sampling() -> bool:
    """
    Run _build_hoc_sample_index on the real HoC file with a small target.
    Verifies stratification and reproducibility without touching the model.
    """
    print(f"\n{'='*60}")
    print("[dry-run] testing HoC stratified sampling")
    print(f"{'='*60}")

    hoc_path = (PROJECT_ROOT / "data" / "experts" / "raw" / "normalized"
                / "uk_house_of_commons.jsonl")

    if not hoc_path.exists():
        print(f"[dry-run] SKIP  HoC file not found at {hoc_path}")
        return True

    try:
        n_sample = 5_000   # small for a fast local test

        idx1 = m._build_hoc_sample_index(hoc_path, n_sample, seed=42)
        idx2 = m._build_hoc_sample_index(hoc_path, n_sample, seed=42)

        assert idx1 == idx2, "sampling is not reproducible — seed is not working"
        print(f"  reproducibility ok (seed=42 gives {len(idx1)} lines both times)")

        # Different seed should give different result
        idx3 = m._build_hoc_sample_index(hoc_path, n_sample, seed=99)
        assert idx1 != idx3, "different seeds produced identical samples — suspicious"
        print(f"  different seeds give different samples ok")

        assert len(idx1) >= n_sample * 0.9, \
            f"sample too small: {len(idx1)} < 90% of {n_sample}"
        assert len(idx1) <= n_sample * 1.2, \
            f"sample too large: {len(idx1)} > 120% of {n_sample}"
        print(f"  sample size {len(idx1)} within expected range of {n_sample}")

        print("[dry-run] PASS  hoc_sampling")
        return True

    except Exception:
        print("[dry-run] FAIL  hoc_sampling")
        traceback.print_exc()
        return False


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results: dict[str, bool] = {}

    # 1. HoC sampling logic — independent of model, tests real HoC file
    results["hoc_sampling"] = test_hoc_sampling()

    # 2. allsides — smallest source, short texts
    #    pass --min-chunk-tokens 30 same as the batch script
    results["allsides"] = run_dry(
        source="allsides",
        max_docs=50,
        extra_cli=["--min-chunk-tokens", "30"],
    )

    # 3. ec_press — full-length press releases, normal chunk settings
    results["ec_press"] = run_dry(
        source="ec_press",
        max_docs=20,
    )

    # 4. hoc — verify the sampled path works end-to-end on 30 speeches
    results["hoc"] = run_dry(
        source="hoc",
        max_docs=30,
        extra_cli=["--hoc-sample-n", "5000", "--min-chunk-tokens", "30"],
    )

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[dry-run] SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("[dry-run] all checks passed — safe to submit to cluster")
        sys.exit(0)
    else:
        print("[dry-run] some checks failed — fix before submitting")
        sys.exit(1)
