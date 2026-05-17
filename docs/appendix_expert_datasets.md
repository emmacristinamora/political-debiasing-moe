# Appendix — Expert Training Datasets

## A.1 Data Statement

The expert training corpora were assembled from nine publicly available English-language text sources spanning legislative speech, government press releases, news media, and online political discussion. No personally identifiable information was collected. All sources are in the public domain or are made available for non-commercial research use. The corpus covers political discourse primarily from the United Kingdom, United States, Ireland, and the European Union, and is therefore most representative of Anglophone, Western democratic political language. Texts are processed as fixed-length chunks of 512 whitespace tokens; chunks shorter than 128 tokens are discarded.

---

## A.2 Corpora

| Source | Label | Type | Geography | Political lean |
|---|---|---|---|---|
| AllSides | `allsides` | News media (balanced) | USA | Mixed |
| European Commission Press Releases | `ec_press` | Government press | EU | Centre-left |
| Irish Government Press Releases | `ire_press` | Government press | Ireland | Centre |
| UK Government Press Releases | `uk_press` | Government press | UK | Mixed |
| UK House of Commons Hansard | `hoc` | Parliamentary speech | UK | Mixed |
| Reddit (Liberal subreddits) | `reddit_liberal` | Social media | USA | Left |
| Reddit (Conservative subreddits) | `reddit_conservative` | Social media | USA | Right |
| US News Media | `us_media` | News media | USA | Mixed |
| US Presidential Speeches | `us_speeches` | Political speech | USA | Mixed |

Sources are grouped into three **families** for analysis and source-diversity checks:
- **Reddit** — `reddit_liberal`, `reddit_conservative`
- **Press** — `allsides`, `ec_press`, `ire_press`, `uk_press`, `us_media`
- **Speeches** — `hoc`, `us_speeches`

---

## A.3 Political Compass Assignment

Each chunk is assigned to one of four quadrants on the political compass using cosine similarity projections onto two steering vectors — one per axis — extracted from Mistral-7B hidden states at layer 20.

| Quadrant | Label | Economic | Social |
|---|---|---|---|
| Q1 | `right_auth` | Right | Authoritarian |
| Q2 | `left_auth` | Left | Authoritarian |
| Q3 | `left_lib` | Left | Libertarian |
| Q4 | `right_lib` | Right | Libertarian |

A chunk is retained if its **confidence margin** (distance between the two axis projections) exceeds **0.015**. This threshold filters out ideologically ambiguous or centrist text that would add noise to expert training.

**Total chunks before filtering:** ~1.6M (all nine corpora combined)
**Total chunks after filtering (conf_margin ≥ 0.015):** 402,633

---

## A.4 Topic Labelling

Each retained chunk is assigned a primary topic label via cosine similarity to nine prototype embeddings:

| Topic | Description |
|---|---|
| `economy` | Taxation, markets, trade, fiscal policy, growth |
| `immigration` | Asylum, border control, citizenship, integration |
| `foreign_policy` | War, alliances, diplomacy, NATO, sanctions |
| `law_order` | Crime, policing, sentencing, criminal justice |
| `environment_energy` | Climate, energy policy, renewables, emissions |
| `culture_identity` | Religion, gender, identity, free speech, values |
| `welfare_labor` | Benefits, unions, workers' rights, social protection |
| `health_education` | Healthcare, schools, universities, public health |
| `governance_institutions` | Elections, parliament, rule of law, accountability |

---

## A.5 Training Split Construction

### Viable topics and held-out topic

Five topics were selected as **viable** (well-represented across all four quadrants): `economy`, `foreign_policy`, `law_order`, `health_education`, `governance_institutions`.

**`immigration`** was held out entirely from training and used as the **topic generalisation validation set** (`val_topic`), testing whether each expert generalises to an unseen political topic.

### Held-out sources (source generalisation)

One source per quadrant was held out from training and reserved as the **source generalisation validation set** (`val_source`):

| Quadrant | Held-out source |
|---|---|
| Q1 right_auth | AllSides |
| Q2 left_auth | UK Gov. Press Releases |
| Q3 left_lib | Irish Gov. Press Releases |
| Q4 right_lib | UK Gov. Press Releases |

### Cell capping

To prevent any single (source × topic) cell from dominating a quadrant's training data, each cell is capped at **250 chunks**. Additional constraints:
- No single source may exceed **70%** of a quadrant's training set
- No single source family (Reddit / Press / Speeches) may exceed **65%**
- No single topic may exceed **50%**
- Chunk length: 50–700 tokens

### Final split sizes

| Expert | Train | Val (in-dist) | Val (source) | Val (topic) | **Total** |
|---|---|---|---|---|---|
| Q1 right_auth | 1,304 | 362 | 921 | 256 | **2,843** |
| Q2 left_auth | 2,057 | 359 | 886 | 938 | **4,240** |
| Q3 left_lib | 5,118 | 929 | 2,086 | 13,555 | **21,688** |
| Q4 right_lib | 3,669 | 629 | 6,018 | 7,042 | **17,358** |
| **Total** | **12,148** | **2,279** | **9,911** | **21,791** | **46,129** |

The large `val_topic` sizes for Q3 and Q4 reflect the heavy immigration content in `left_lib` and `right_lib` corpora (primarily Reddit and UK gov. press releases).

---

## A.6 Dataset Imbalance

Q1 (`right_auth`) is substantially smaller than the other quadrants (2,843 vs. 17,358–21,688). This reflects the underlying corpus structure: the press and parliamentary speech corpora (which dominate the dataset) contain relatively little text that is simultaneously economically right-wing and socially authoritarian. Reddit Conservative contributes the largest share of Q1 training data (66%). This imbalance is not corrected by oversampling, as doing so would amplify the stylistic idiosyncrasies of a single source.

---

## A.7 Figures

**Figure A.1** — Training and validation chunks per expert, broken down by source family (Reddit, Press, Speeches). Q3 and Q4 are larger primarily because Reddit Conservative subreddits produce a large volume of right-libertarian and left-libertarian text.

**Figure A.2** — Retained chunk count by topic and quadrant (log-scale heatmap). Economy, governance, and law & order are the most consistently represented topics across all quadrants.

**Figure A.3** — Source share per quadrant in the final training set. The cell cap prevents any single source from exceeding 70% of a quadrant, ensuring stylistic diversity within each expert.
