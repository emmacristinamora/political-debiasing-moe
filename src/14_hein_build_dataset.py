# src/14_hein_build_dataset.py

# Tier 2 — external ground-truth validation of the steering vectors, step 1/3.
#
# Builds the labeled dataset: per-legislator speech corpora from the hein-daily
# Congressional Record, each joined to its DW-NOMINATE ideal point (Voteview).
# A legislator's speeches are pooled across the Congress window through the
# stable Voteview ICPSR id, then subsampled to a common word budget so corpus
# size cannot confound the later projection.
#
#   step 1  src/14_hein_build_dataset.py     -> legislator_dataset.jsonl
#   step 2  src/15_hein_project_compass.py   -> compass coordinates  (GPU)
#   step 3  src/16_hein_dwnominate_analysis.py -> correlation report


# === IMPORTS ===

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HEIN_DIR = PROJECT_ROOT / "data" / "external" / "hein-daily"
DEFAULT_VOTEVIEW = PROJECT_ROOT / "data" / "external" / "HSall_members.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "hein_dwnominate"
DEFAULT_CONGRESSES = [110, 111, 112, 113, 114]

# Voteview party_code -> compact label
PARTY_LABELS = {"100": "D", "200": "R", "328": "I"}
# hein chamber letter -> Voteview chamber word
CHAMBER_WORDS = {"H": "House", "S": "Senate"}


@dataclass
class SpeakerUnit:
    """All speeches by one speaker within one Congress (a hein speakerid)."""

    congress: int
    speaker_id: str
    last_name: str
    first_name: str
    chamber: str          # "H" or "S"
    state: str
    speeches: list[str] = field(default_factory=list)
    word_count: int = 0


@dataclass
class Legislator:
    """One person pooled across the window, joined to a DW-NOMINATE score."""

    icpsr: str
    bioname: str
    party: str
    chambers: list[str]
    states: list[str]
    congresses: list[int]
    nominate_dim1: float
    nominate_dim2: float
    speeches: list[str] = field(default_factory=list)


# === HELPERS: IO ===

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tier 2 step 1 — build the per-legislator DW-NOMINATE-labeled dataset."
    )
    parser.add_argument("--hein-dir", type=Path, default=DEFAULT_HEIN_DIR)
    parser.add_argument("--voteview", type=Path, default=DEFAULT_VOTEVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--congresses", type=int, nargs="+", default=DEFAULT_CONGRESSES)
    parser.add_argument("--min-speech-words", type=int, default=50,
                        help="drop speeches shorter than this (procedural boilerplate).")
    parser.add_argument("--target-words", type=int, default=5000,
                        help="common per-legislator word budget after equalizing.")
    parser.add_argument("--max-words-per-unit", type=int, default=15000,
                        help="cap stored words per (congress, speaker) to bound memory.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a pretty-printed JSON summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# === HELPERS: HEIN PARSING ===

def parse_speaker_map(path: Path) -> dict[str, dict[str, str]]:
    """
    Parse a hein NNN_SpeakerMap.txt into {speech_id: speaker fields}.

    The SpeakerMap only lists speeches confidently attributed to a real member,
    so unattributed procedural speech ("The CLERK", ...) is dropped here for free.
    """
    if not path.is_file():
        raise FileNotFoundError(f"speaker map not found: {path}")
    speech_to_speaker: dict[str, dict[str, str]] = {}
    with path.open(encoding="latin-1") as handle:
        next(handle)  # header: speakerid|speech_id|lastname|firstname|chamber|state|...
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 9:
                continue
            speech_to_speaker[parts[1]] = {
                "speaker_id": parts[0],
                "last_name": parts[2].strip().upper(),
                "first_name": parts[3].strip().upper(),
                "chamber": parts[4].strip(),
                "state": parts[5].strip().upper(),
            }
    return speech_to_speaker


def collect_speaker_units(
    hein_dir: Path, congresses: list[int], min_speech_words: int, max_words_per_unit: int
) -> dict[tuple[int, str], SpeakerUnit]:
    """
    Stream the speeches files and accumulate speeches per (congress, speaker).

    Logic:
        For each Congress, the SpeakerMap gives the speaker of every attributed
        speech; the speeches file is streamed line by line and each speech long
        enough to clear the procedural-boilerplate threshold is appended to its
        speaker's unit, until that unit reaches the per-unit word cap.
    """
    units: dict[tuple[int, str], SpeakerUnit] = {}
    for congress in congresses:
        speaker_map = parse_speaker_map(hein_dir / f"{congress}_SpeakerMap.txt")
        speeches_path = hein_dir / f"speeches_{congress}.txt"
        if not speeches_path.is_file():
            raise FileNotFoundError(f"speeches file not found: {speeches_path}")

        kept = 0
        with speeches_path.open(encoding="latin-1") as handle:
            next(handle)  # header: speech_id|speech
            for line in handle:
                split = line.rstrip("\n").split("|", 1)
                if len(split) != 2:
                    continue
                speech_id, text = split
                speaker = speaker_map.get(speech_id)
                if speaker is None:
                    continue
                n_words = len(text.split())
                if n_words < min_speech_words:
                    continue

                key = (congress, speaker["speaker_id"])
                unit = units.get(key)
                if unit is None:
                    unit = SpeakerUnit(
                        congress=congress,
                        speaker_id=speaker["speaker_id"],
                        last_name=speaker["last_name"],
                        first_name=speaker["first_name"],
                        chamber=speaker["chamber"],
                        state=speaker["state"],
                    )
                    units[key] = unit
                if unit.word_count >= max_words_per_unit:
                    continue
                unit.speeches.append(text.strip())
                unit.word_count += n_words
                kept += 1
        print(f"  congress {congress}: kept {kept} speeches across "
              f"{sum(1 for k in units if k[0] == congress)} speakers")
    return units


# === HELPERS: DW-NOMINATE JOIN ===

def build_voteview_index(path: Path, congresses: list[int]) -> dict[tuple[int, str, str], list[dict[str, Any]]]:
    """
    Index Voteview members by (congress, chamber, state) for name matching.

    Each member row is parsed into last/first name from its bioname
    ("LASTNAME, Firstname ...") and kept with its icpsr and DW-NOMINATE scores.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Voteview file not found: {path}")
    wanted = set(congresses)
    index: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            congress = int(row["congress"])
            if congress not in wanted or not row["nominate_dim1"]:
                continue
            bioname = row["bioname"]
            last, _, rest = bioname.partition(", ")
            member = {
                "icpsr": row["icpsr"],
                "bioname": bioname,
                "last_name": last.strip().upper(),
                "first_name": rest.strip().upper(),
                "party": PARTY_LABELS.get(row["party_code"], "I"),
                "nominate_dim1": float(row["nominate_dim1"]),
                "nominate_dim2": float(row["nominate_dim2"]) if row["nominate_dim2"] else 0.0,
            }
            index[(congress, row["chamber"], row["state_abbrev"])].append(member)
    return index


def match_member(unit: SpeakerUnit, index: dict[tuple[int, str, str], list[dict[str, Any]]]) -> dict[str, Any] | None:
    """
    Match one hein speaker to a Voteview member row.

    Logic:
        Look up candidates sharing the (congress, chamber, state) key, keep
        those whose last name matches exactly, and disambiguate any remaining
        tie by first-name initial. Returns None when the match is not unique.
    """
    chamber_word = CHAMBER_WORDS.get(unit.chamber)
    if chamber_word is None:
        return None
    candidates = index.get((unit.congress, chamber_word, unit.state), [])
    by_last = [m for m in candidates if m["last_name"] == unit.last_name]
    if len(by_last) == 1:
        return by_last[0]
    if len(by_last) > 1 and unit.first_name:
        initial = unit.first_name[0]
        by_first = [m for m in by_last if m["first_name"][:1] == initial]
        if len(by_first) == 1:
            return by_first[0]
    return None


def build_legislators(
    units: dict[tuple[int, str], SpeakerUnit],
    index: dict[tuple[int, str, str], list[dict[str, Any]]],
) -> tuple[dict[str, Legislator], int]:
    """
    Match every speaker unit to DW-NOMINATE and pool units by ICPSR person.

    Returns (legislators_by_icpsr, n_unmatched_units). DW-NOMINATE scores are
    averaged over the Congresses in which the person appears.
    """
    by_icpsr: dict[str, Legislator] = {}
    dim_samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    unmatched = 0
    for unit in units.values():
        member = match_member(unit, index)
        if member is None:
            unmatched += 1
            continue
        icpsr = member["icpsr"]
        legislator = by_icpsr.get(icpsr)
        if legislator is None:
            legislator = Legislator(
                icpsr=icpsr,
                bioname=member["bioname"],
                party=member["party"],
                chambers=[],
                states=[],
                congresses=[],
                nominate_dim1=0.0,
                nominate_dim2=0.0,
            )
            by_icpsr[icpsr] = legislator
        legislator.speeches.extend(unit.speeches)
        if unit.chamber not in legislator.chambers:
            legislator.chambers.append(unit.chamber)
        if unit.state not in legislator.states:
            legislator.states.append(unit.state)
        if unit.congress not in legislator.congresses:
            legislator.congresses.append(unit.congress)
        dim_samples[icpsr].append((member["nominate_dim1"], member["nominate_dim2"]))

    for icpsr, legislator in by_icpsr.items():
        samples = dim_samples[icpsr]
        legislator.nominate_dim1 = sum(d1 for d1, _ in samples) / len(samples)
        legislator.nominate_dim2 = sum(d2 for _, d2 in samples) / len(samples)
        legislator.congresses.sort()
    return by_icpsr, unmatched


# === HELPERS: EQUALIZE ===

def equalize_corpus(speeches: list[str], target_words: int, rng: random.Random) -> tuple[str, int] | None:
    """
    Subsample to an exact common word budget.

    Speeches are shuffled and added until the budget is reached, then the
    concatenation is truncated to exactly target_words so corpus size is
    identical across legislators and cannot confound the projection. Returns
    (text, n_speeches) or None when the legislator has too little text.
    """
    order = speeches[:]
    rng.shuffle(order)
    selected: list[str] = []
    words = 0
    for speech in order:
        selected.append(speech)
        words += len(speech.split())
        if words >= target_words:
            break
    if words < target_words:
        return None
    text = " ".join(" ".join(selected).split()[:target_words])
    return text, len(selected)


# === MAIN ===

def main() -> None:
    """Build and write the per-legislator DW-NOMINATE-labeled speech dataset."""
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"collecting speeches from Congresses {args.congresses}")
    units = collect_speaker_units(
        args.hein_dir, args.congresses, args.min_speech_words, args.max_words_per_unit
    )
    print(f"speaker units: {len(units)}")

    print("matching to Voteview DW-NOMINATE")
    index = build_voteview_index(args.voteview, args.congresses)
    legislators, unmatched = build_legislators(units, index)
    print(f"matched persons: {len(legislators)} | unmatched units: {unmatched}")

    rows: list[dict[str, Any]] = []
    dropped_short = 0
    for legislator in legislators.values():
        equalized = equalize_corpus(legislator.speeches, args.target_words, rng)
        if equalized is None:
            dropped_short += 1
            continue
        text, n_speeches = equalized
        rows.append({
            "icpsr": legislator.icpsr,
            "bioname": legislator.bioname,
            "party": legislator.party,
            "chambers": legislator.chambers,
            "states": legislator.states,
            "congresses": legislator.congresses,
            "nominate_dim1": legislator.nominate_dim1,
            "nominate_dim2": legislator.nominate_dim2,
            "n_speeches": n_speeches,
            "n_words": args.target_words,
            "text": text,
        })
    rows.sort(key=lambda r: r["nominate_dim1"])

    dataset_path = args.output_dir / "legislator_dataset.jsonl"
    report_path = args.output_dir / "build_report.json"
    write_jsonl(rows, dataset_path)

    parties = sorted({r["party"] for r in rows})
    report = {
        "congresses": args.congresses,
        "min_speech_words": args.min_speech_words,
        "target_words": args.target_words,
        "n_speaker_units": len(units),
        "n_unmatched_units": unmatched,
        "n_matched_persons": len(legislators),
        "n_dropped_below_target": dropped_short,
        "n_legislators": len(rows),
        "party_counts": {p: sum(1 for r in rows if r["party"] == p) for p in parties},
        "nominate_dim1_range": [rows[0]["nominate_dim1"], rows[-1]["nominate_dim1"]] if rows else [],
    }
    save_json(report, report_path)

    print(f"\nlegislators in dataset : {len(rows)}")
    print(f"party balance          : {report['party_counts']}")
    print(f"dropped below target   : {dropped_short}")
    print(f"wrote {dataset_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
