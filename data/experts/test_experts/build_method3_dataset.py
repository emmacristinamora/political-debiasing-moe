"""
build_method3_dataset.py

Builds the full method 3 (consistency) evaluation dataset from OpinionQA.
Targets 15 questions per topic across 20 topic categories.
Run from project root:
    python3 data/experts/test_experts/build_method3_dataset.py
"""

import json
import csv
import ast
import os
import numpy as np
from collections import defaultdict

MODEL_INPUT_DIR  = "/Users/stefi/Downloads/model_input"
HUMAN_RESP_DIR   = "/Users/stefi/Downloads/human_resp"
OUTPUT_PATH      = "data/experts/test_experts/methode_3_data.jsonl"
TARGET_PER_TOPIC = 15

WAVE_FILES = [
    "Pew_American_Trends_Panel_W26.csv",
    "Pew_American_Trends_Panel_W27.csv",
    "Pew_American_Trends_Panel_W29.csv",
    "Pew_American_Trends_Panel_W32.csv",
    "Pew_American_Trends_Panel_W34.csv",
    "Pew_American_Trends_Panel_W36.csv",
    "Pew_American_Trends_Panel_W41.csv",
    "Pew_American_Trends_Panel_W42.csv",
    "Pew_American_Trends_Panel_W43.csv",
    "Pew_American_Trends_Panel_W45.csv",
    "Pew_American_Trends_Panel_W49.csv",
    "Pew_American_Trends_Panel_W50.csv",
    "Pew_American_Trends_Panel_W54.csv",
    "Pew_American_Trends_Panel_W82.csv",
    "Pew_American_Trends_Panel_W92.csv",
]

RELEVANT_COARSE_TOPICS = {
    "economy and inequality":                         "economic",
    "political issues":                               "both",
    "healthcare system":                              "economic",
    "immigration":                                    "social",
    "discrimination":                                 "social",
    "gender & sexuality":                             "social",
    "science":                                        "both",
    "religion":                                       "social",
    "global attitudes and foreign policy":            "both",
    "crime/security":                                 "social",
    "corporations, banks, technology and automation": "economic",
    "education":                                      "economic",
    "future":                                         "both",
    "race":                                           "social",
    "news, social media, data, privacy":              "social",
    "community health":                               "economic",
    "relationships and family":                       "social",
    "self-perception and values":                     "both",
    "leadership":                                     "both",
    "job/career":                                     "economic",
}

EXPLICIT_EXTRAS = {
    "religion": [
        ("RELIG_GOV_W92",     "W92", "social"),
        ("GOODEVIL_W92",      "W92", "social"),
        ("SOCIETY_SSM_W92",   "W92", "social"),
        ("MARRFAM_W92",       "W92", "social"),
        ("USEXCEPT_W92",      "W92", "social"),
        ("COMPROMISEVAL_W92", "W92", "social"),
        ("GOVWASTE_W92",      "W92", "social"),
        ("POLICY3MOD_W92",    "W92", "social"),
        ("CNTRYFAIR_W92",     "W92", "social"),
        ("PPLRESP_W92",       "W92", "social"),
    ],
    "immigration": [
        ("IL_IMM_PRI_W92",    "W92", "social"),
        ("LEGALIMMIGAMT_W92", "W92", "social"),
        ("UNIMMIGCOMM_W92",   "W92", "social"),
        ("RACESURV52MOD_W92", "W92", "social"),
        ("OPENIDEN_W92",      "W92", "social"),
        ("INEQ5_l_W54",       "W54", "social"),
        ("INEQ5_m_W54",       "W54", "social"),
        ("GOVPRIORITYb_W54",  "W54", "social"),
        ("GAP21Q33_g_W82",    "W82", "social"),
        ("GAP21Q33_s_W82",    "W82", "social"),
        ("INEQ5_i_W54",       "W54", "social"),
        ("GAP21Q46_W82",      "W82", "social"),
        ("GAP21Q47_W82",      "W82", "social"),
    ],
    "future": [
        ("GAP21Q2_W82",   "W82", "both"),
        ("GAP21Q24_W82",  "W82", "both"),
        ("GAP21Q27_W82",  "W82", "both"),
        ("GAP21Q28_W82",  "W82", "both"),
        ("FUTURE_W42",    "W42", "both"),
        ("PAST_W42",      "W42", "both"),
        ("GAP21Q9_W82",   "W82", "both"),
        ("GAP21Q11_W82",  "W82", "both"),
        ("SUPERPWR_W92",  "W92", "both"),
        ("LIFEFIFTY_W92", "W92", "both"),
        ("PREDICTA_W27",  "W27", "both"),
        ("PREDICTB_W27",  "W27", "both"),
        ("PREDICTC_W27",  "W27", "both"),
        ("PREDICTD_W27",  "W27", "both"),
        ("GAP21Q14_W82",  "W82", "both"),
    ],
    "self-perception and values": [
        ("ECONFAIR_W92",    "W92", "both"),
        ("WOMENOBS_W92",    "W92", "both"),
        ("BUSPROFIT_W92",   "W92", "both"),
        ("GOVPROTCT_W92",   "W92", "both"),
        ("GOVAID_W92",      "W92", "both"),
        ("POORASSIST_W92",  "W92", "both"),
        ("ELITEUNDMOD_W92", "W92", "both"),
        ("CANQUALPOL_W92",  "W92", "both"),
        ("VTRGHTPRIV1_W92", "W92", "both"),
        ("ALLIES_W92",      "W92", "both"),
        ("PEACESTR_W92",    "W92", "both"),
    ],
    "job/career": [
        ("ROBJOB5D_W27",  "W27", "economic"),
        ("ROBJOB6_W27",   "W27", "economic"),
        ("ROBJOB7_W27",   "W27", "economic"),
        ("WORK2_W27",     "W27", "economic"),
        ("WORK4A_W27",    "W27", "economic"),
        ("WORK4B_W27",    "W27", "economic"),
        ("WORK4C_W27",    "W27", "economic"),
        ("WORRY2b_W54",   "W54", "economic"),
        ("JOBTRAIN_W54",  "W54", "economic"),
        ("ROBJOB3A_W27",  "W27", "economic"),
        ("ROBJOB3B_W27",  "W27", "economic"),
    ],
    "education": [
        ("INSTN_CLGS_W92",    "W92", "economic"),
        ("FREECOLL_W92",      "W92", "economic"),
        ("SOCIETY_JBCLL_W92", "W92", "economic"),
        ("GOVRESP_c_W54",     "W54", "economic"),
        ("GOVRESP_d_W54",     "W54", "economic"),
        ("INSTN_K12_W92",     "W92", "economic"),
        ("GOVRESP_h_W54",     "W54", "economic"),
    ],
    "leadership": [
        ("CANDEXP_W92",    "W92", "both"),
        ("CANMTCHPOL_W92", "W92", "both"),
        ("REPRSNTREP_W92", "W92", "both"),
        ("REPRSNTDEM_W92", "W92", "both"),
        ("CANQUALPOL_W92", "W92", "both"),
        ("GAP21Q17_W82",   "W82", "both"),
        ("GAP21Q29_W82",   "W82", "both"),
        ("GAP21Q30_W82",   "W82", "both"),
        ("GAP21Q31_W82",   "W82", "both"),
        ("DIFFPARTY_W92",  "W92", "both"),
    ],
    "discrimination": [
        ("PROG_RRETRO_W92",   "W92", "social"),
        ("PROG_RNEED_W92",    "W92", "social"),
        ("PROG_RNEED2b_W92",  "W92", "social"),
        ("WHADVANT_W92",      "W92", "social"),
        ("SOCIETY_RHIST_W92", "W92", "social"),
        ("SOCIETY_WHT_W92",   "W92", "social"),
        ("GAP21Q23_W82",      "W82", "social"),
        ("GAP21Q20_W82",      "W82", "social"),
        ("POLINTOL2_a_W92",   "W92", "social"),
        ("POLINTOL2_b_W92",   "W92", "social"),
    ],
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_all_waves():
    all_q = {}
    for fname in WAVE_FILES:
        path = os.path.join(MODEL_INPUT_DIR, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found")
            continue
        wave = fname.replace("Pew_American_Trends_Panel_", "").replace(".csv", "")
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = row.get("key", "").strip()
                q   = row.get("question", "").strip()
                try:
                    opts = ast.literal_eval(row.get("options", "[]"))
                except Exception:
                    opts = []
                if key and q:
                    all_q[key] = {"key": key, "question": q,
                                  "options": opts, "wave": wave}
    print(f"Loaded {len(all_q)} questions from {len(WAVE_FILES)} waves")
    return all_q


def is_ordinal(question, options):
    non_ref = [o for o in options if "refused" not in o.lower()]
    if len(non_ref) < 2 or len(non_ref) > 6:
        return False
    keywords = ["agree", "important", "likely", "often", "good", "safe",
                "worry", "confident", "approve", "support", "positive",
                "favorable", "concern", "problem", "difficult", "easy",
                "better", "worse", "much", "lot", "satisfied", "priority",
                "responsibility", "favor", "oppose", "should", "would"]
    text = question.lower() + " " + " ".join(str(o) for o in options).lower()
    return any(kw in text for kw in keywords)


def make_record(key, q_data, coarse_topic, fine_topics, axis):
    return {
        "key":            key,
        "question":       q_data["question"],
        "options":        q_data["options"],
        "coarse_topic":   coarse_topic,
        "fine_topics":    fine_topics,
        "axis_relevance": axis,
        "wave":           q_data["wave"],
    }

# ── main ──────────────────────────────────────────────────────────────────────

print("Loading topic mapping ...")
topic_map = np.load(
    os.path.join(HUMAN_RESP_DIR, "topic_mapping.npy"),
    allow_pickle=True
).item()
print(f"  {len(topic_map)} entries in topic mapping")

print("Loading wave CSVs ...")
all_q = load_all_waves()

prefix_to_key = {v["question"][:80]: k for k, v in all_q.items()}

# ── pass 1: topic-mapping-driven matching ─────────────────────────────────────
print("\nPass 1: topic-mapping matching ...")
questions    = []
added_keys   = set()
topic_counts = defaultdict(int)

for q_text, tinfo in topic_map.items():
    coarse_topics = tinfo.get("cg", [])
    fine_topics   = tinfo.get("fg", [])

    matched_topic = None
    axis = None
    for ct in coarse_topics:
        for rel_topic, rel_axis in RELEVANT_COARSE_TOPICS.items():
            if rel_topic in ct.lower() or ct.lower() in rel_topic:
                matched_topic = rel_topic
                axis = rel_axis
                break
        if matched_topic:
            break
    if not matched_topic:
        continue
    if topic_counts[matched_topic] >= TARGET_PER_TOPIC:
        continue

    matched_key = None
    prefix = q_text.strip()[:80]
    if prefix in prefix_to_key:
        matched_key = prefix_to_key[prefix]
    else:
        for k, v in all_q.items():
            if q_text.strip()[:60] in v["question"] or v["question"][:60] in q_text.strip():
                matched_key = k
                break
    if not matched_key or matched_key in added_keys:
        continue

    q_data = all_q[matched_key]
    if not is_ordinal(q_data["question"], q_data["options"]):
        continue

    questions.append(make_record(matched_key, q_data, matched_topic, fine_topics, axis))
    added_keys.add(matched_key)
    topic_counts[matched_topic] += 1

print(f"  {len(questions)} questions after pass 1")

# ── pass 2: explicit extras ───────────────────────────────────────────────────
print("\nPass 2: explicit extras ...")
added_explicit = 0
for coarse_topic, extras in EXPLICIT_EXTRAS.items():
    for key, wave_hint, axis in extras:
        if topic_counts[coarse_topic] >= TARGET_PER_TOPIC:
            break
        if key in added_keys:
            continue
        if key not in all_q:
            print(f"  [WARN] {key} not found in loaded waves")
            continue
        q_data = all_q[key]
        non_ref = [o for o in q_data["options"] if "refused" not in o.lower()]
        if len(non_ref) < 2:
            continue
        questions.append(make_record(key, q_data, coarse_topic, [coarse_topic], axis))
        added_keys.add(key)
        topic_counts[coarse_topic] += 1
        added_explicit += 1
        print(f"  Added {key} -> {coarse_topic} ({topic_counts[coarse_topic]}/{TARGET_PER_TOPIC})")

print(f"  {added_explicit} questions added in pass 2")

# ── save ──────────────────────────────────────────────────────────────────────
output = {
    "meta": {
        "description": (
            "OpinionQA-based evaluation set for method 3 (consistency). "
            "Tests whether each expert maintains ideological alignment "
            "across diverse topic categories."
        ),
        "total":            len(questions),
        "topics_covered":   len(topic_counts),
        "target_per_topic": TARGET_PER_TOPIC,
        "usage": (
            "Feed each question to each adapter with no persona prompt. "
            "Record responses across 20-30 runs. Check whether each adapter "
            "maintains its assigned quadrant position consistently across "
            "all topic categories."
        ),
        "axis_relevance_key": {
            "economic": "primarily tests economic left-right axis",
            "social":   "primarily tests libertarian-authoritarian axis",
            "both":     "tests both axes",
        },
    },
    "topic_counts": dict(topic_counts),
    "questions":    questions,
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"Total questions: {len(questions)}")
print("\nFinal topic counts:")
for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
    status = "OK" if count >= TARGET_PER_TOPIC else "CLOSE" if count >= 10 else "LOW"
    print(f"  [{status}] {topic}: {count}")
