import json
import csv
import ast
import os
import numpy as np
from collections import defaultdict

MODEL_INPUT_DIR = "/Users/stefi/Downloads/model_input"
HUMAN_RESP_DIR = "/Users/stefi/Downloads/human_resp"

RELEVANT_COARSE_TOPICS = {
    "economy and inequality": "economic",
    "political issues": "both",
    "healthcare system": "economic",
    "immigration": "social",
    "discrimination": "social",
    "gender & sexuality": "social",
    "science": "both",
    "religion": "social",
    "global attitudes and foreign policy": "both",
    "crime/security": "social",
    "corporations, banks, technology and automation": "economic",
    "education": "economic",
    "future": "both",
    "race": "social",
    "news, social media, data, privacy": "social",
    "community health": "economic",
    "relationships and family": "social",
    "self-perception and values": "both",
    "leadership": "both",
    "job/career": "economic",
}

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

print("Loading topic mapping...")
topic_mapping = np.load(
    os.path.join(HUMAN_RESP_DIR, "topic_mapping.npy"),
    allow_pickle=True
).item()
print(f"Topic mapping loaded: {len(topic_mapping)} entries")

print("Loading wave questions...")
all_questions = {}
for wave_file in WAVE_FILES:
    path = os.path.join(MODEL_INPUT_DIR, wave_file)
    if not os.path.exists(path):
        continue
    wave_name = wave_file.replace("Pew_American_Trends_Panel_", "").replace(".csv", "")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = row.get("key", "").strip()
            question = row.get("question", "").strip()
            options_raw = row.get("options", "[]").strip()
            try:
                options = ast.literal_eval(options_raw)
            except:
                options = []
            if key and question:
                all_questions[key] = {
                    "key": key,
                    "question": question,
                    "options": options,
                    "wave": wave_name
                }
print(f"Questions loaded: {len(all_questions)} total")

print("Matching questions to topics...")
matched = []
topic_counts = defaultdict(int)

for question_text, topic_info in topic_mapping.items():
    coarse_topics = topic_info.get("cg", [])
    fine_topics = topic_info.get("fg", [])
    
    matched_topic = None
    axis = None
    for ct in coarse_topics:
        ct_lower = ct.lower()
        for relevant_topic, relevant_axis in RELEVANT_COARSE_TOPICS.items():
            if relevant_topic in ct_lower or ct_lower in relevant_topic:
                matched_topic = ct
                axis = relevant_axis
                break
        if matched_topic:
            break
    
    if not matched_topic:
        continue
    
    # Find the key for this question in the wave files
    matched_key = None
    for key, q_data in all_questions.items():
        if question_text.strip()[:60] in q_data["question"] or q_data["question"][:60] in question_text.strip():
            matched_key = key
            break
    
    # Filter: only keep questions with ordinal options (agree/disagree or similar scales)
    options = []
    if matched_key:
        options = all_questions[matched_key]["options"]
    else:
        continue
    
    # Check options are ordinal and not too many choices
    ordinal_keywords = ["agree", "important", "likely", "often", "good", "safe", 
                        "worry", "confident", "approve", "support", "positive",
                        "favorable", "concern", "problem", "difficult", "easy"]
    has_refused = "Refused" in options or "refused" in str(options).lower()
    non_refused_options = [o for o in options if "refused" not in o.lower()]
    
    is_ordinal = (
        len(non_refused_options) >= 2 and 
        len(non_refused_options) <= 5 and
        any(kw in question_text.lower() or any(kw in str(o).lower() for o in options) 
            for kw in ordinal_keywords)
    )
    
    if not is_ordinal:
        continue
    
    if topic_counts[matched_topic] >= 15:
        continue
    
    matched.append({
        "key": matched_key,
        "question": question_text,
        "options": options,
        "coarse_topic": matched_topic,
        "fine_topics": fine_topics,
        "axis_relevance": axis,
        "wave": all_questions[matched_key]["wave"] if matched_key else "unknown"
    })
    topic_counts[matched_topic] += 1

print(f"\nMatched {len(matched)} questions across topics:")
for topic, count in sorted(topic_counts.items()):
    print(f"  {topic}: {count} questions")

output = {
    "meta": {
        "description": "OpinionQA-based evaluation set for method 3 (consistency). Tests whether each expert maintains ideological alignment across diverse topic categories.",
        "total": len(matched),
        "topics_covered": len(topic_counts),
        "usage": "Feed each question to each adapter with no persona prompt. Record responses across 20-30 runs. Check whether each adapter maintains its assigned quadrant position consistently across all topic categories.",
        "axis_relevance_key": {
            "economic": "primarily tests economic left-right axis",
            "social": "primarily tests libertarian-authoritarian axis",
            "both": "tests both axes"
        }
    },
    "topic_counts": dict(topic_counts),
    "questions": matched
}

output_path = "data/experts/test_experts/methode 3 data.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {output_path}")
print(f"Total questions: {len(matched)}")
print(f"Topics covered: {len(topic_counts)}")