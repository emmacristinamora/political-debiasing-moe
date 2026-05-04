# data/experts/normalize_corpora.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOWNLOADS_DIR_DEFAULT = Path.home() / "Downloads"

NORMALIZED_DIR = PROJECT_ROOT / "data" / "experts" / "raw" / "normalized"
MANIFESTS_DIR  = PROJECT_ROOT / "data" / "experts" / "raw" / "manifests"

# AllSides contains article excerpts (~60-90 words each), not full articles.
# HoC contains individual speeches, many of which are brief interjections.
# The rest are full documents where 100 words is reasonable.
VALID_PARTIES = {"D", "R"}
MIN_WORDS = 100
MAX_WORDS = 800
MIN_WORD_COUNT_BY_SOURCE: dict[str, int] = {
    "allsides":            30,
    "reddit_liberal":      100,
    "reddit_conservative": 100,
    "hoc":                 30,
    "ec_press":            100,
    "uk_press":            100,
    "ire_press":           100,
    "hein_congressional":  100,
}

VALID_SOURCES = {
    "allsides",
    "reddit_liberal",
    "reddit_conservative",
    "hoc",
    "ec_press",
    "uk_press",
    "ire_press",
    "hein_congressional",
}

SOURCE_FILENAMES: dict[str, str] = {
    "allsides":            "allsides_balanced_news_headlines-texts.csv",
    "reddit_liberal":      "Liberal.json",
    "reddit_conservative": "Conservative.json",
    "hoc":                 "Corp_HouseOfCommons_V2.rds",
    "ec_press":            "EC-PressReleases_1985-2020_clean.RDS",
    "uk_press":            "UK-GovPressReleases.Rds",
    "ire_press":           "IRE-GovPressReleases.Rds",
    "hein_congressional":  "hein-daily",
}

OUTPUT_FILENAMES: dict[str, str] = {
    "allsides":            "allsides.jsonl",
    "reddit_liberal":      "reddit_liberal.jsonl",
    "reddit_conservative": "reddit_conservative.jsonl",
    "hoc":                 "uk_house_of_commons.jsonl",
    "ec_press":            "ec_press_releases.jsonl",
    "uk_press":            "uk_gov_press_releases.jsonl",
    "ire_press":           "ire_gov_press_releases.jsonl",
    "hein_congressional":  "hein_congressional.jsonl",
}

# RDS files that pyreadr cannot handle due to encoding — fall back to R subprocess
RDS_NEEDS_R_SUBPROCESS: set[str] = {"hoc", "ec_press"}


# === SCHEMA ===

@dataclass
class Document:
    document_id: str
    text: str
    source_name: str
    source_family: str
    language: str
    raw_dataset: str
    title: str | None = None
    date: str | None = None
    speaker_or_author: str | None = None
    twitter_flag: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id":       self.document_id,
            "text":              self.text,
            "source_name":       self.source_name,
            "source_family":     self.source_family,
            "language":          self.language,
            "title":             self.title,
            "date":              self.date,
            "speaker_or_author": self.speaker_or_author,
            "twitter_flag":      self.twitter_flag,
            "raw_dataset":       self.raw_dataset,
            "metadata":          self.metadata,
        }


# === CLI ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw expert corpora into a canonical JSONL schema."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=sorted(VALID_SOURCES) + ["all"],
        help="Source to normalize. Pass 'all' to run every source sequentially.",
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=DOWNLOADS_DIR_DEFAULT,
        help=f"Directory containing raw downloaded files (default: {DOWNLOADS_DIR_DEFAULT}).",
    )
    return parser.parse_args()


# === TEXT UTILITIES ===

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def is_acceptable(text: str, min_words: int) -> bool:
    return bool(text) and word_count(text) >= min_words


def short_hash(value: str, length: int = 8) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:length]


def parse_tags_field(raw: str | None) -> list[str]:
    """Parse AllSides tags stored as stringified Python lists."""
    if not raw or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if t]
    except (ValueError, SyntaxError):
        pass
    return [t.strip() for t in raw.split(",") if t.strip()]


def utc_timestamp_to_iso(ts: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def coerce_date(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nat", "none", "null", "nan"):
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    return None


def coerce_str(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "nat") else None


# === IO ===

def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rds_pyreadr(path: Path) -> pd.DataFrame:
    try:
        import pyreadr  # type: ignore
    except ImportError as exc:
        raise ImportError("pyreadr is required: pip install pyreadr") from exc
    result = pyreadr.read_r(str(path))
    df = result[None]
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame from {path}, got {type(df)}")
    return df


def load_rds_via_r(path: Path) -> pd.DataFrame:
    """Export RDS to a temp CSV via R subprocess, then read with pandas."""
    if not _r_available():
        raise RuntimeError(
            f"R is required to read {path.name} but 'Rscript' was not found on PATH."
        )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    r_script = (
        f"df <- readRDS('{path}')\n"
        f"write.csv(df, '{tmp_path}', row.names=FALSE, fileEncoding='UTF-8')\n"
    )

    print(f"[rds] Exporting {path.name} via R → temp CSV (may take a moment for large files)...")
    result = subprocess.run(
        ["Rscript", "--vanilla", "-"],
        input=r_script,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R subprocess failed for {path.name}:\n{result.stderr}")

    df = pd.read_csv(tmp_path, low_memory=False)
    tmp_path.unlink(missing_ok=True)
    print(f"[rds] Loaded {len(df):,} rows from {path.name}")
    return df


def _r_available() -> bool:
    return subprocess.run(["which", "Rscript"], capture_output=True).returncode == 0


def write_jsonl(docs: Iterator[Document], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# === NORMALIZERS ===

def load_procedural_phrases(aux_dir: Path) -> set[str]:
    path = aux_dir / "procedural.txt"
    if not path.exists():
        print(f"[hein] WARNING: procedural.txt not found at {path} — skipping filter")
        return set()
    phrases: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("|")
            if parts:
                phrases.add(parts[0].strip().lower())
    print(f"[hein] Loaded {len(phrases):,} procedural phrases")
    return phrases
 
 
def load_speaker_map(hein_dir: Path, session: int) -> dict[str, dict[str, str]]:
    """
    Returns a dict mapping speech_id -> {party, chamber, state, lastname, firstname}
    Only keeps D and R parties.
    """
    fname = f"{session:03d}_SpeakerMap.txt"
    path = hein_dir / fname
    if not path.exists():
        print(f"[hein] WARNING: {fname} not found — skipping session {session}")
        return {}
 
    speaker_map: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        next(f)  # skip header: speakerid|speech_id|lastname|firstname|chamber|state|gender|party|district|nonvoting
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 8:
                continue
            speech_id = parts[1].strip()
            party     = parts[7].strip().upper()
            if party not in VALID_PARTIES:
                continue
            speaker_map[speech_id] = {
                "party":     party,
                "chamber":   parts[4].strip(),   # S or H
                "state":     parts[5].strip(),
                "lastname":  parts[2].strip(),
                "firstname": parts[3].strip(),
            }
    return speaker_map
 
 
def is_procedural(text: str, procedural_phrases: set[str]) -> bool:
    if not procedural_phrases:
        return False
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in procedural_phrases)
 
 
# ── NORMALIZER ────────────────────────────────────────────────────────────────
 
def normalize_hein_session(
    hein_dir: Path,
    session: int,
    procedural_phrases: set[str],
) -> Iterator[Document]:
    """Yield normalized Documents for one congressional session."""
 
    speeches_file = hein_dir / f"speeches_{session:03d}.txt"
    if not speeches_file.exists():
        print(f"[hein] WARNING: speeches_{session:03d}.txt not found — skipping")
        return
 
    speaker_map = load_speaker_map(hein_dir, session)
    if not speaker_map:
        print(f"[hein] WARNING: no valid speaker map for session {session} — skipping")
        return
 
    print(f"[hein] Session {session:03d}: "
          f"{len(speaker_map):,} mapped speakers, reading speeches ...")
 
    stats = {"total": 0, "no_speaker": 0, "procedural": 0,
             "too_short": 0, "too_long": 0, "written": 0}
 
    with speeches_file.open("r", encoding="utf-8", errors="replace") as f:
        next(f)  # skip header: speech_id|speech
        for line in f:
            stats["total"] += 1
            parts = line.strip().split("|", 1)
            if len(parts) < 2:
                stats["no_speaker"] += 1
                continue
 
            speech_id = parts[0].strip()
            raw_text  = parts[1].strip()
 
            # Must have speaker metadata with valid party
            if speech_id not in speaker_map:
                stats["no_speaker"] += 1
                continue
 
            speaker = speaker_map[speech_id]
            text    = normalize_whitespace(raw_text)
            wc      = word_count(text)
 
            if wc < MIN_WORDS:
                stats["too_short"] += 1
                continue
            if wc > MAX_WORDS:
                stats["too_long"] += 1
                continue
            if is_procedural(text, procedural_phrases):
                stats["procedural"] += 1
                continue
 
            party   = speaker["party"]
            chamber = speaker["chamber"]
            state   = speaker["state"]
            name    = f"{speaker['firstname']} {speaker['lastname']}".strip()
 
            # Chamber label
            chamber_full = "Senate" if chamber == "S" else "House"
 
            # Party label for metadata
            party_full = "Democrat" if party == "D" else "Republican"
 
            document_id = f"hein_{session:03d}_{speech_id}"
 
            yield Document(
                document_id       = document_id,
                text              = text,
                source_name       = "Hein_CongressionalSpeeches",
                source_family     = "institutional_speech",
                language          = "en",
                raw_dataset       = f"hein-daily_session_{session:03d}",
                speaker_or_author = name if name.strip() else None,
                metadata          = {
                    "source_url":      None,
                    "bias_rating":     None,
                    "tags":            [],
                    "party":           party_full,
                    "party_code":      party,
                    "chamber":         chamber_full,
                    "state":           state,
                    "session":         session,
                    "speech_id":       speech_id,
                    "subreddit":       None,
                    "doc_type":        "speech",
                    "word_count":      wc,
                },
            )
            stats["written"] += 1
 
    print(f"[hein] Session {session:03d} done | "
          f"total={stats['total']:,} "
          f"no_speaker={stats['no_speaker']:,} "
          f"procedural={stats['procedural']:,} "
          f"too_short={stats['too_short']:,} "
          f"too_long={stats['too_long']:,} "
          f"written={stats['written']:,}")

def normalize_allsides(downloads_dir: Path) -> Iterator[Document]:
    path = downloads_dir / SOURCE_FILENAMES["allsides"]
    min_words = MIN_WORD_COUNT_BY_SOURCE["allsides"]
    rows = load_csv(path)
    print(f"[allsides] Loaded {len(rows):,} rows")

    skipped = 0
    for idx, row in enumerate(rows):
        raw_text = coerce_str(row.get("text", ""))
        if not raw_text:
            skipped += 1
            continue
        text = normalize_whitespace(raw_text)
        if not is_acceptable(text, min_words):
            skipped += 1
            continue

        document_id = f"allsides_{idx:07d}"
        tags = parse_tags_field(row.get("tags"))

        yield Document(
            document_id=document_id,
            text=text,
            source_name="AllSides",
            source_family="news_article",
            language="en",
            raw_dataset="allsides_balanced_news_headlines-texts",
            title=coerce_str(row.get("heading") or row.get("title")),
            metadata={
                "source_url":  None,
                "bias_rating": coerce_str(row.get("bias_rating")),
                "tags":        tags,
                "party":       None,
                "subreddit":   None,
                "doc_type":    "article",
                "outlet":      coerce_str(row.get("source")),
            },
        )

    print(f"[allsides] Skipped {skipped:,} short/empty rows")


def normalize_reddit(downloads_dir: Path, side: str) -> Iterator[Document]:
    assert side in ("liberal", "conservative")
    key = f"reddit_{side}"
    path = downloads_dir / SOURCE_FILENAMES[key]
    min_words = MIN_WORD_COUNT_BY_SOURCE[key]
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path.name}, got {type(data)}")
    print(f"[reddit_{side}] Loaded {len(data):,} rows")

    prefix = "reddit_lib" if side == "liberal" else "reddit_con"
    skipped = 0

    for idx, row in enumerate(data):
        raw_text = coerce_str(row.get("articles", ""))
        if not raw_text:
            skipped += 1
            continue
        text = normalize_whitespace(raw_text)
        if not is_acceptable(text, min_words):
            skipped += 1
            continue

        url = coerce_str(row.get("urls", ""))
        document_id = f"{prefix}_{short_hash(url or str(idx))}_{idx:07d}"
        date = utc_timestamp_to_iso(row.get("created_utc"))

        yield Document(
            document_id=document_id,
            text=text,
            source_name="RedditIdeologicalArticles",
            source_family="article_opinion",
            language="en",
            raw_dataset="reddit_ideological_extreme_bias",
            date=date,
            metadata={
                "source_url":      url,
                "bias_rating":     None,
                "tags":            [],
                "party":           None,
                "subreddit":       None,
                "community_label": side,
                "flair":           coerce_str(row.get("flair")),
                "url_domain":      coerce_str(row.get("url_domain")),
                "num_upvotes":     row.get("num_upvotes"),
                "num_comments":    row.get("num_comments"),
                "doc_type":        "article",
            },
        )

    print(f"[reddit_{side}] Skipped {skipped:,} short/empty rows")


def normalize_hoc(downloads_dir: Path) -> Iterator[Document]:
    path = downloads_dir / SOURCE_FILENAMES["hoc"]
    min_words = MIN_WORD_COUNT_BY_SOURCE["hoc"]
    df = load_rds_via_r(path)
    print(f"[hoc] Processing {len(df):,} rows")

    skipped = 0
    for idx, row in df.iterrows():
        raw_text = coerce_str(row.get("text", ""))
        if not raw_text:
            skipped += 1
            continue
        text = normalize_whitespace(raw_text)
        if not is_acceptable(text, min_words):
            skipped += 1
            continue

        date_str = coerce_date(row.get("date"))
        seq = str(idx).zfill(7)
        document_id = f"hoc_{date_str or 'nodate'}_{seq}"

        yield Document(
            document_id=document_id,
            text=text,
            source_name="UK_HouseOfCommons_V2",
            source_family="institutional_speech",
            language="en",
            raw_dataset="Corp_HouseOfCommons_V2",
            date=date_str,
            speaker_or_author=coerce_str(row.get("speaker")),
            metadata={
                "source_url":   None,
                "bias_rating":  None,
                "tags":         [],
                "party":        coerce_str(row.get("party")),
                "subreddit":    None,
                "doc_type":     "speech",
                "agenda":       coerce_str(row.get("agenda")),
                "speechnumber": row.get("speechnumber"),
                "chair":        bool(row.get("chair")),
                "terms":        row.get("terms"),
                "parliament":   coerce_str(row.get("parliament")),
                "iso3country":  coerce_str(row.get("iso3country")),
            },
        )

    print(f"[hoc] Skipped {skipped:,} short/empty rows")


def normalize_ec_press(downloads_dir: Path) -> Iterator[Document]:
    path = downloads_dir / SOURCE_FILENAMES["ec_press"]
    min_words = MIN_WORD_COUNT_BY_SOURCE["ec_press"]
    df = load_rds_via_r(path)
    print(f"[ec_press] Processing {len(df):,} rows")

    skipped = 0
    for idx, row in df.iterrows():
        raw_text = coerce_str(row.get("text", ""))
        if not raw_text:
            skipped += 1
            continue
        text = normalize_whitespace(raw_text)
        if not is_acceptable(text, min_words):
            skipped += 1
            continue

        ipnum = coerce_str(row.get("ipnum", ""))
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", ipnum) if ipnum else str(idx).zfill(7)
        document_id = f"ec_{safe_id}"
        date_str = coerce_date(row.get("date"))

        yield Document(
            document_id=document_id,
            text=text,
            source_name="EC_PressReleases",
            source_family="institutional_press_release",
            language="en",
            raw_dataset="EC-PressReleases_1985-2020",
            title=coerce_str(row.get("title")),
            date=date_str,
            metadata={
                "source_url":  None,
                "bias_rating": None,
                "tags":        [],
                "party":       None,
                "subreddit":   None,
                "doc_type":    "press_release",
                "institution": "European Commission",
                "ipnum":       ipnum,
                "nchars":      row.get("nchars"),
                "ntokens":     row.get("ntokens"),
                "nsentences":  row.get("nsentences"),
            },
        )

    print(f"[ec_press] Skipped {skipped:,} short/empty rows")


def normalize_uk_press(downloads_dir: Path) -> Iterator[Document]:
    path = downloads_dir / SOURCE_FILENAMES["uk_press"]
    min_words = MIN_WORD_COUNT_BY_SOURCE["uk_press"]
    df = load_rds_pyreadr(path)
    print(f"[uk_press] Processing {len(df):,} rows")

    skipped = 0
    for idx, row in df.iterrows():
        raw_text = coerce_str(row.get("text", ""))
        if not raw_text:
            skipped += 1
            continue
        text = normalize_whitespace(raw_text)
        if not is_acceptable(text, min_words):
            skipped += 1
            continue

        url = coerce_str(row.get("url", ""))
        document_id = f"uk_press_{short_hash(url or str(idx))}_{str(idx).zfill(7)}"
        date_str = coerce_date(row.get("date"))
        is_speech = bool(row.get("speech", False))

        yield Document(
            document_id=document_id,
            text=text,
            source_name="UK_GovPressReleases",
            source_family="institutional_speech" if is_speech else "institutional_press_release",
            language="en",
            raw_dataset="UK-GovPressReleases",
            title=coerce_str(row.get("headline")),
            date=date_str,
            speaker_or_author=coerce_str(row.get("author")),
            metadata={
                "source_url":  url,
                "bias_rating": None,
                "tags":        [],
                "party":       None,
                "subreddit":   None,
                "doc_type":    "speech" if is_speech else "press_release",
                "institution": "UK Government",
            },
        )

    print(f"[uk_press] Skipped {skipped:,} short/empty rows")


def normalize_ire_press(downloads_dir: Path) -> Iterator[Document]:
    path = downloads_dir / SOURCE_FILENAMES["ire_press"]
    min_words = MIN_WORD_COUNT_BY_SOURCE["ire_press"]
    df = load_rds_pyreadr(path)
    print(f"[ire_press] Processing {len(df):,} rows")

    skipped = 0
    for idx, row in df.iterrows():
        raw_text = coerce_str(row.get("text", ""))
        if not raw_text:
            skipped += 1
            continue
        text = normalize_whitespace(raw_text)
        if not is_acceptable(text, min_words):
            skipped += 1
            continue

        url = coerce_str(row.get("url", ""))
        document_id = f"ire_press_{short_hash(url or str(idx))}_{str(idx).zfill(7)}"
        date_str = coerce_date(row.get("date"))

        yield Document(
            document_id=document_id,
            text=text,
            source_name="IRE_GovPressReleases",
            source_family="institutional_press_release",
            language="en",
            raw_dataset="IRE-GovPressReleases",
            title=coerce_str(row.get("headline")),
            date=date_str,
            speaker_or_author=coerce_str(row.get("author")),
            metadata={
                "source_url":  url,
                "bias_rating": None,
                "tags":        [],
                "party":       None,
                "subreddit":   None,
                "doc_type":    "press_release",
                "institution": "Irish Government",
            },
        )

    print(f"[ire_press] Skipped {skipped:,} short/empty rows")


# === DISPATCH ===

def get_normalizer_iterator(source: str, downloads_dir: Path) -> Iterator[Document]:
    if source == "allsides":
        return normalize_allsides(downloads_dir)
    if source == "reddit_liberal":
        return normalize_reddit(downloads_dir, "liberal")
    if source == "reddit_conservative":
        return normalize_reddit(downloads_dir, "conservative")
    if source == "hoc":
        return normalize_hoc(downloads_dir)
    if source == "ec_press":
        return normalize_ec_press(downloads_dir)
    if source == "uk_press":
        return normalize_uk_press(downloads_dir)
    if source == "ire_press":
        return normalize_ire_press(downloads_dir)
    if source == "hein_congressional":
        return normalize_hein_congressional(downloads_dir)
    raise ValueError(f"Unknown source: {source}")


# === MANIFEST ===

def update_manifest(
    source: str,
    input_path: Path,
    output_path: Path,
    n_written: int,
) -> None:
    manifest_path = MANIFESTS_DIR / "raw_file_inventory.json"
    manifest: dict[str, Any] = {}

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            try:
                manifest = json.load(handle)
            except json.JSONDecodeError:
                manifest = {}

    manifest[source] = {
        "raw_file":        str(input_path),
        "normalized_file": str(output_path),
        "n_documents":     n_written,
        "min_word_count":  MIN_WORD_COUNT_BY_SOURCE[source],
        "normalized_at":   datetime.now(tz=timezone.utc).isoformat(),
    }

    save_json(manifest, manifest_path)
    print(f"[manifest] Updated {manifest_path}")
    

def normalize_hein_congressional(downloads_dir: Path) -> Iterator[Document]:
    hein_dir = downloads_dir / "hein-daily"
    procedural_phrases = load_procedural_phrases(hein_dir)
    sessions = list(range(97, 115))
    for session in sessions:
        yield from normalize_hein_session(hein_dir, session, procedural_phrases)


# === ORCHESTRATION ===

def run_source(source: str, downloads_dir: Path) -> None:
    input_path  = downloads_dir / SOURCE_FILENAMES[source]
    output_path = NORMALIZED_DIR / OUTPUT_FILENAMES[source]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected raw file for source '{source}' at: {input_path}"
        )

    print(f"\n{'='*60}")
    print(f"[start] source={source}")
    print(f"[io]    input  → {input_path}")
    print(f"[io]    output → {output_path}")
    print(f"{'='*60}\n")

    doc_iter = get_normalizer_iterator(source, downloads_dir)
    n_written = write_jsonl(doc_iter, output_path)

    update_manifest(
        source=source,
        input_path=input_path,
        output_path=output_path,
        n_written=n_written,
    )

    print(f"\n{'='*60}")
    print(f"[done] source={source} | documents written={n_written:,}")
    print(f"[done] output → {output_path}")
    print(f"{'='*60}\n")


# === MAIN ===

def main() -> None:
    args = parse_args()
    downloads_dir = args.downloads_dir.expanduser().resolve()

    if not downloads_dir.exists():
        print(f"[error] Downloads directory not found: {downloads_dir}", file=sys.stderr)
        sys.exit(1)

    sources_to_run = sorted(VALID_SOURCES) if args.source == "all" else [args.source]

    for source in sources_to_run:
        run_source(source=source, downloads_dir=downloads_dir)


if __name__ == "__main__":
    main()
