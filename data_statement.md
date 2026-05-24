# Data Statement — Expert Training Corpora

This document records the provenance, licensing, preprocessing, and known limitations of every dataset used to build the four quadrant expert corpora. All sources were normalized to a canonical JSONL schema via `data/experts/normalize_corpora.py` before any downstream processing.

---

## Overview

The expert training corpora span three medium categories, chosen deliberately to prevent the experts from learning medium-specific style rather than political orientation.

| Medium | Sources |
|---|---|
| Social media | Reddit (Liberal, Conservative) |
| Press and media | AllSides, US News Articles (NYT / WSJ / NYP), EU Commission, UK Government, Irish Government press releases |
| Speeches | UK House of Commons, US Presidential Speeches |

---

## 1. Reddit Ideological and Extreme Bias Dataset

**Source:** Mendeley Data
**URL:** https://data.mendeley.com/datasets/2tdr9sjd83/3
**License:** CC BY 4.0
**Format:** JSON (two files: `Liberal.json`, `Conservative.json`)
**Coverage:** Reddit posts from r/Liberal and r/Conservative subreddits

**Provenance:** Published as part of a dataset collection for studying ideological bias in social media. Contains posts collected from two politically-aligned subreddits. The liberal and conservative labels derive directly from subreddit membership; no additional labeling was applied.

**Fields used:** `articles` (post text), `urls` (URL), `created_utc` (timestamp), `flair`, `url_domain`, `num_upvotes`, `num_comments`

**Preprocessing applied:**
- Minimum word count: 100 words
- Texts below threshold discarded; no other content filtering
- Document IDs constructed as `reddit_lib_{hash8}_{idx:07d}` / `reddit_con_{hash8}_{idx:07d}` where `hash8` is the first 8 characters of the MD5 hash of the URL

**Known limitations:** Subreddit membership is self-selected; posts may not represent the full spectrum within each labeled ideology. reddit_conservative is by far the largest source in the dataset (~690K raw documents), creating a significant size imbalance relative to other sources.

---

## 2. AllSides Balanced News Headlines and Texts

**Source:** Qbias project, GitHub
**URL:** https://github.com/irgroup/Qbias/blob/main/allsides_balanced_news_headlines-texts.csv
**License:** See Qbias repository (data derived from AllSides.com)
**Format:** CSV
**Coverage:** News article excerpts labeled with AllSides media bias ratings (Left, Center, Right)

**Provenance:** Compiled by the Information Retrieval Group (irgroup) as part of the Qbias query bias project. Texts are excerpts from news articles curated by AllSides, which independently rates media outlets for political bias.

**Fields used:** `text`, `heading`/`title`, `bias_rating`, `tags`, `source`

**Preprocessing applied:**
- Minimum word count: 30 words (articles are short excerpts, typically 60–90 words; standard 100-word threshold would discard most of the corpus)
- `--min-chunk-tokens 30` applied in script 05 for the same reason
- Document IDs constructed as `allsides_{idx:07d}`

**Known limitations:** Excerpts are short. AllSides bias ratings are outlet-level, not article-level; individual articles may not perfectly reflect the outlet's rated orientation. The bias label is used only for provenance tracking; it is not used in training target construction.

---

## 3. US News Articles (New York Times, Wall Street Journal, New York Post)

**Source:** Factiva (Bocconi University subscription); collection and preprocessing conducted as part of a prior research project on U.S. political discourse
**Reference:** Mora, E. C. (2025). *Political Discourse Analysis in U.S. Presidential Debates and Media (1960–2024)*. GitHub: https://github.com/emmacristinamora/thesis
**Format:** CSV (derived from Factiva RTF exports)
**Coverage:** 1,024 articles across three U.S. news outlets, four election years (2012, 2016, 2020, 2024), three topic areas (foreign policy and national security, healthcare and public health, immigration and borders)

**Provenance:** Articles were retrieved via Factiva using structured keyword queries. Search filters applied at collection time:
- Article type: news articles only (debates and transcripts excluded)
- Subject: Politics/International Relations AND Elections
- Region: United States
- Language: English
- Date: January of each election year (2012, 2016, 2020, 2024)
- Outlets selected based on AllSides media bias ratings:
  - *New York Times* (left-leaning; chosen for Factiva coverage quality and article length)
  - *Wall Street Journal* (center-right)
  - *New York Post* (right-leaning; Fox News and Washington Examiner were unavailable on Factiva at collection time)
- Per query: first 30 results by relevance (exception: 2012, NYP, immigration — 18 articles found)

**Collection pipeline (thesis repository):**
1. `media_parse_factiva.py` — parsed Factiva RTF exports into individual articles using regex on Factiva field markers (HD, TD); extracted metadata from filenames (outlet, year, topic)
2. `media_dataset_cleaning.py` — removed Factiva artefacts (TAG tokens, tail phrases); assigned outlet_leaning labels (D / NaN / R)
3. `media_final_adjustments.py` — removed remaining boilerplate via manually-specified regex patterns; deduplicated exact duplicates
4. `03_media_topic_modeling.ipynb` — chunked articles (~250 words, 50-word overlap); embedded with SBERT (all-MiniLM-L6-v2); assigned debate-aligned topic labels via cosine similarity to debate theme centroids; applied confidence filtering (theme cosine ≥ 0.45, margin ≥ 0.05); balanced to 225 chunks per outlet

**Final dataset after balancing:** 675 chunks (225 × NYT, 225 × WSJ, 225 × NYP)

**Preprocessing applied in this project:**
- Minimum word count: 64 words (`--min-chunk-tokens 64` in script 05; articles are already pre-chunked)
- Document IDs constructed from outlet and article index

**Known limitations:** NYP is a tabloid with short, politically charged articles; its style differs substantially from NYT and WSJ, which may conflate outlet register with political orientation. Collection was limited to January of each election year for budget and access reasons; coverage is therefore not representative of year-round political discourse. Fox News and Washington Examiner were unavailable on Factiva. This source covers only US English-language media.

---

## 4. EU Commission Press Releases

**Source:** Harvard Dataverse
**URL:** https://dataverse.harvard.edu/file.xhtml?fileId=6562387&version=1.1
**License:** See Harvard Dataverse entry
**Format:** RDS (loaded via R subprocess due to pyreadr encoding incompatibility)
**Coverage:** European Commission press releases, 1985–2020

**Provenance:** Compiled as part of a comparative political language corpus at Harvard Dataverse. Press releases are official institutional communications from the European Commission.

**Fields used:** `text` (press release body), `ipnum` (document identifier), date metadata

**Preprocessing applied:**
- Minimum word count: 100 words
- Document IDs constructed as `ec_{ipnum_sanitized}` (non-alphanumeric characters → `_`)
- R subprocess timeout: 600 seconds

**Known limitations:** Press releases represent official European institutional language. Political orientation is primarily center-liberal by the nature of the institution; this source contributes to the left-auth and left-lib quadrant pools but offers limited right-leaning signal.

---

## 5. UK Government Press Releases

**Source:** Harvard Dataverse
**URL:** https://dataverse.harvard.edu/file.xhtml?fileId=6562442&version=1.1
**License:** See Harvard Dataverse entry
**Format:** RDS (loaded via pyreadr)
**Coverage:** UK Government press releases

**Provenance:** Compiled as part of the same comparative institutional language corpus as the EU Commission data. Covers official press releases from UK government departments and ministries.

**Fields used:** `text`, `url`, `speech` (boolean flag indicating whether entry is a speech rather than a press release)

**Preprocessing applied:**
- Minimum word count: 100 words
- Source family assigned as `institutional_speech` if `speech == True`, else `institutional_press_release`
- Document IDs constructed as `uk_press_{hash8}_{idx:07d}` (MD5 of URL)

**Known limitations:** Political orientation of UK Government press releases varies with the governing party (Conservative or Labour) and the period covered. No party metadata is included in the normalized schema; party-period signal is not preserved downstream.

---

## 6. Irish Government Press Releases

**Source:** Harvard Dataverse
**URL:** https://dataverse.harvard.edu/file.xhtml?fileId=6562326&version=1.1
**License:** See Harvard Dataverse entry
**Format:** RDS (loaded via pyreadr)
**Coverage:** Irish Government press releases

**Provenance:** Part of the same Harvard Dataverse comparative corpus. Official communications from the Irish government.

**Fields used:** `text`, `url`

**Preprocessing applied:**
- Minimum word count: 100 words
- Document IDs constructed as `ire_press_{hash8}_{idx:07d}` (MD5 of URL)

**Known limitations:** Irish political discourse does not map cleanly onto the two-axis Political Compass, particularly the economic axis. The EU Commission, UK, and Irish press releases were chosen together to provide European institutional language as a counterweight to the Reddit and US speech sources.

---

## 7. UK House of Commons Parliamentary Speeches

**Source:** Harvard Dataverse
**URL:** https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/L4OAKN/W2SVMF&version=1.0
**License:** See Harvard Dataverse entry
**Format:** RDS (loaded via R subprocess due to pyreadr encoding incompatibility)
**Coverage:** UK House of Commons speeches (full corpus: ~1.6M speeches)

**Provenance:** Compiled from Hansard (the official parliamentary record of the UK House of Commons). Includes speeches from MPs across all parties.

**Fields used:** speech text, date, speaker metadata

**Preprocessing applied:**
- Minimum word count: 30 words (many HoC contributions are brief procedural interjections; the standard 100-word threshold would discard the majority of the corpus)
- `--min-chunk-tokens 30` applied in script 05
- **Stratified sampling:** 200,000 speeches drawn from the full ~1.6M corpus; strata defined by decade × party, proportional allocation with minimum 1 per stratum, seed=42. This is applied in script 05 at scoring time, not at normalization time.
- Document IDs constructed as `hoc_{date_or_nodate}_{idx:07d}`

**Known limitations:** The stratified sample of 200,000 is still the largest single source after Reddit conservative (~690K chunks). UK parliamentary language covers Conservative and Labour as primary parties, which map onto the right-auth and left-auth quadrants respectively but do not cover the libertarian dimension directly. Brief interjections (procedural language, "hear hear", order calls) pass the 30-word threshold and add noise.

---

## 8. US Presidential Speeches

**Source:** Miller Center, University of Virginia
**URL:** https://data.millercenter.org/
**License:** See Miller Center terms
**Format:** Plain text files, one per speech
**Coverage:** US presidential speeches from 1985 to 2026 (254 speeches); earlier speeches excluded

**Provenance:** The Miller Center archives presidential speeches, addresses, and press conferences. The 1985 cutoff was applied to keep the language temporally consistent with the other sources in the corpus and to focus on the post-Reagan era of modern US political discourse.

**Fields used:** speech text, date, speaker (president)

**Preprocessing applied:**
- Minimum word count: 100 words (all presidential speeches are well above this threshold)
- Document IDs constructed from speaker and date
- No `--min-chunk-tokens` override (default 128 tokens applied in script 05)

**Known limitations:** All speeches from 1985–2026 come from sitting US presidents; the corpus therefore alternates between Democratic (left-leaning) and Republican (right-leaning) administrations depending on the period. No within-corpus balancing by president or party was applied.

---

## Normalization Pipeline

All sources pass through `data/experts/normalize_corpora.py` before any scoring or training. The normalization pipeline applies the following steps identically for every source:

1. Load the raw file in full (CSV, JSON, or RDS via pyreadr or R subprocess)
2. Extract the text field; drop rows where the field is missing or empty
3. Normalize whitespace: `\r\n`/`\r` → `\n`; runs of spaces/tabs → single space; 3+ consecutive newlines → 2 newlines; strip leading/trailing whitespace
4. Apply source-specific minimum word count filter (`len(text.split()) < threshold` → drop); maximum 800 words for all sources
5. Construct a stable document ID from source prefix + sequential index or MD5 hash of URL
6. Map all source-specific field names to the canonical schema
7. Write to JSONL incrementally (no full corpus held in memory)
8. Record per-source statistics in `data/experts/raw/manifests/raw_file_inventory.json`

**Canonical Document schema:** `document_id`, `text`, `source_name`, `source_family`, `language`, `raw_dataset`, `title`, `date`, `speaker_or_author`, `twitter_flag`, `metadata`

The word count thresholds were set by manual inspection of each corpus and reflect the nature of the source. They are conservative: only rows clearly too short to carry political signal are dropped. Thresholds and document counts are logged in `data/experts/raw/manifests/raw_file_inventory.json`.

---

## Summary Table

| Source | Medium | Language | Coverage | Min words | License |
|---|---|---|---|---|---|
| Reddit (Liberal, Conservative) | Social media | English | r/Liberal, r/Conservative | 100 | CC BY 4.0 |
| AllSides | News excerpts | English | Multi-outlet, various dates | 30 | See Qbias repo |
| US News (NYT/WSJ/NYP) | News articles | English | 2012–2024 election years | 64 | Factiva (institutional) |
| EU Commission press releases | Institutional | English | 1985–2020 | 100 | See Dataverse |
| UK Government press releases | Institutional | English | Various | 100 | See Dataverse |
| Irish Government press releases | Institutional | English | Various | 100 | See Dataverse |
| UK House of Commons | Speeches | English | Full corpus (~1.6M); 200K sampled | 30 | See Dataverse |
| US Presidential Speeches | Speeches | English | 1985–2026 | 100 | See Miller Center |
