# CLAUDE.md — Final Project Implementation Playbook

> **Auto-loaded into every Claude Code session opened from this folder.**
> Keep this file lean. If a section grows past ~30 lines, split it out and link.
> Update the **Status** block at the bottom of each working session.

---

## 1. Identity

- **Course:** CSGY-6513 Big Data, Section D, Term 2 — Prof. Amit Patel
- **Project:** "The Language of Complaints" — NLP on NYC 311 service requests
- **Team:** George Gideon Sale (gs4602), Aayush Prranav Chandrashekar (ac11929), Shreeram Sankar (ss18731)
- **Submitting member (Brightspace):** gs4602
- **Build mode:** **solo build by gs4602**, three names on cover. Code style and commit history reflect a single author.
- **Source of truth for scope:** [`../Our Project Proposal/proposal.md`](../Our%20Project%20Proposal/proposal.md). If reality forces a deviation, log it in §11 and flag for the user. The "Changes in Technology" section of the final report (rubric expectation #4) is the legitimate place to document deliberate pivots.

---

## 2. Hard Constraints (read before every task)

1. **Live 15-min Zoom demo. No-show = zero on entire project.** Demo URL must be shareable so any teammate can present.
2. **Demoed dataset must be ≥ 1M rows.** Working sample is **2M stratified**; full corpus 43M is referenced as "we built to scale, sampled for iteration speed."
3. **Single live interface.** No platform juggling during demo. One Streamlit URL does everything.
4. **Anti-AI-detection writing rules** for any text seen by graders (proposal, report, slides, code comments, reflections):
   - Student tone, not corporate. Slightly imperfect is fine.
   - **No em-dashes.** No fancy unicode (arrows, smart quotes, ellipses). Plain ASCII only: `-`, `to`, `...`
   - Reflections in paragraphs, not bullet lists. Under 3 short paragraphs each.
5. **Code-quality rubric (5 pts):** every non-trivial line gets a comment that *sounds like a student*.
6. **Don't drift onto the report or slides** until user explicitly says so. Implementation first.
7. **Read [`Prompt.txt`](../Prompt.txt) before starting every numbered task.** User instruction, very strict.
8. **Ask before deciding** anything user-visible (tooling pivot, scope cut, demo path). Always offer a recommended option.

---

## 3. Locked Decisions (2026-05-02)

| Decision | Value | Why |
|---|---|---|
| **Compute** | **Google Colab Pro** (user has subscription) | Free GPU for BERT, no quota grief, full pip ecosystem, 24h sessions on Pro |
| **Spark mode** | **Local mode** (`spark.master("local[*]")`) | 2M rows fits comfortably in Colab Pro RAM; still real PySpark DataFrame + MLlib |
| **Storage during dev** | Parquet on Drive mount + Delta Lake via `delta-spark` pip | Persistent across Colab sessions |
| **UI framework** | **Streamlit** | 3-5x less boilerplate than Dash; perfect for paste-text → see-prediction widgets |
| **Deployment / shareable URL** | **Streamlit Community Cloud** primary, **HuggingFace Spaces** fallback | Public persistent URL, deploy from GitHub, free |
| **Inference path** | **Pure-Python at serve time** (no Spark in deployed app) | SCC has 1GB limit and no Java; export Spark models to numpy/sklearn-compatible artifacts |
| **Repo** | **GitHub public** under `gs4602` | Required for shareable URL deployment |
| **Sample size** | **2M rows stratified by Problem** | Above the 1M floor with margin; trains in <10 min on Colab |
| **Language** | Python 3.11 | |
| **Dashboard branding** | NYU purple accents, otherwise Streamlit default | |

---

## 4. Grading Rubric Map (30 pts total)

| Pts | Item | Deliverable | Done when... |
|----:|------|-------------|--------------|
| 5 | Presentation | Slide deck + script | Deferred until user signal |
| 5 | Q & A | Team prep | Deferred |
| 5 | Code Quality | Notebooks + scripts | All cells run top-to-bottom on a fresh kernel; every block has a student-tone comment |
| 10 | Live Demo | Streamlit app at a public URL | Single command brings up everything in <3 min; demo flow rehearsed |
| 5 | Report + Code submission | PDF + GitHub URL | Deferred (report) / repo cleaned at end |

**Demo is 33% of grade. Optimize the build for demoability.**

---

## 5. End-to-End Workflow

Each phase has a "Done when" gate. Don't move past a gate without it green.

### Phase 0 — Environment lock-in
- Confirm Colab Pro session works; verify Drive mount; install pinned deps from `requirements.txt`.
- One Spark cell creates a `SparkSession` in local mode and reads 10K rows from SODA API.
- **Done when:** `notebooks/00_env_check.ipynb` runs end-to-end on Colab; schema printed.

### Phase 1 — Ingest + normalize *(P0)*
- Pull both 311 datasets via SODA API (incremental, paginated).
- Normalize column names across the two halves (2020+ renamed "Complaint Type" -> "Problem", "Descriptor" -> "Problem Detail").
- Write to Parquet on Drive, partitioned by year.
- Build a 2M stratified sample via `sampleBy`.
- **Done when:** sample on disk, `df.count() == 2_000_000`, schema committed.

### Phase 2 — Text preprocessing *(P0)*
- NLTK tokenizer, stopword removal, lemmatization. Bundle NLTK data files into the repo.
- Wrap as a Spark `Transformer` so it slots into a `Pipeline`.
- **Done when:** preprocessing runs in <2 min on dev sample; `tokens` column non-null in 99th percentile.

### Phase 3 — Classification *(P0)*
- TF-IDF + Logistic Regression and Random Forest on top-20 Problem labels.
- Stratified 80/20 split via `sampleBy` (NOT `randomSplit`).
- Baselines: majority-class predictor + keyword heuristic.
- **Done when:** macro-F1 >= 0.75 on top 20 categories; `PipelineModel` serialized; portable export saved (numpy coefficients for LR + sklearn-compatible RF).

### Phase 4 — Resolution time regression *(P0)*
- Features: TF-IDF (capped vocab) + Agency + Borough + hour-of-day + day-of-week.
- Model: Linear Regression + Random Forest Regressor; metric MAE (hours).
- Baseline: median-per-category.
- **Done when:** model beats baseline by >=10% MAE; portable export saved.

### Phase 5 — Word2Vec + clustering *(P1)*
- Train Word2Vec (vector size 100, window 5).
- Aggregate doc embeddings; K-Means with k swept 5-30; pick by silhouette.
- **Done when:** silhouette curve plotted, top-10 terms per cluster committed, KeyedVectors exported.

### Phase 6 — Geographic + Census join *(P1)*
- 311 -> community districts (GeoJSON spatial join via Shapely) -> ACS demographics.
- Per-district complaint vocabulary fingerprint table.
- **Done when:** fingerprint table on disk; per-district top-10 distinctive terms saved.

### Phase 7 — Streamlit dashboard *(P1, central deliverable)*
- Five tabs in `dashboard/app.py`:
  1. **Triage Bot** (centerpiece): paste complaint -> classify + predict resolution + cluster + nearest neighbors + map pin
  2. **City Pulse**: Folium choropleth + time series of complaint volume
  3. **Cluster Atlas**: Word2Vec clusters with top terms + per-district word clouds
  4. **Bias Audit**: per-borough macro-F1 and resolution-time bias chart, citing Kontokosta & Hong (2021)
  5. **Pipeline Status**: dataset stats, model versions, training metadata
- Sidebar: borough filter, date range, top-K controls, "Refresh from SODA API" button
- **Done when:** `streamlit run app.py` works locally and remote URL is live.

### Phase 8 — Streaming PoC *(P2)*
- Spark Structured Streaming with `Trigger.AvailableNow`.
- Watch a folder, classify rows with the trained `PipelineModel`, sink to Delta.
- A "Live Pulse" tab in Streamlit subscribes to the Delta sink and displays last-N classified rows.
- **Done when:** dropping a CSV into watch folder produces classified output visible in the dashboard within one micro-batch.

### Phase 9 — LDA + trend detection *(P2)*
- LDA k=20-30; manual labeling of top topics.
- Sliding-window term-frequency over time; flag spikes.
- **Done when:** at least one interpretable topic and one spike with date range documented.

### Phase 10 — BERT embedding sidebar *(P2 novelty)*
- Encode the 2M sample (or stratified sub-sample) with `sentence-transformers/all-MiniLM-L6-v2`.
- Re-run K-Means on BERT embeddings; show side-by-side vs Word2Vec in Cluster Atlas tab.
- **Done when:** comparison plot visible; cached embeddings parquet on disk.

### Phase 11 — Submission packaging *(deferred until user signal)*
- Hardening pass on README, requirements, demo script. Report PDF authored last.

---

## 6. Single-Interface Demo Concept

```
Streamlit app at https://<repo>.streamlit.app
  Sidebar:
    - Borough filter (multi-select)
    - Date range slider (2010 to 2025)
    - Top-K controls
    - [Refresh from SODA API] button (P2)
  Tabs:
    [Triage Bot]   <- DEFAULT TAB, demo centerpiece
    [City Pulse]
    [Cluster Atlas]
    [Bias Audit]
    [Live Pulse]    (P2 streaming PoC)
    [Pipeline Status]
```

The Triage Bot tab is the demo's emotional peak: prof types a complaint -> sees classification + resolution prediction + cluster + map + similar past complaints, all in <2 seconds.

---

## 7. File / Folder Conventions

```
FINAL PROJECT IMPLEMENTATION/
├── CLAUDE.md                            # this file
├── HADOOP_ASSIGNMENT_MASTER_PROMPT.md
├── README.md                            # project README; deploy entry point
├── requirements.txt                     # pip pins
├── .gitignore
├── PROFESSOR DOCS/                      # rubric, expectations
│
├── docs/
│   └── STEP_BY_STEP_GUIDE.md            # micro-detailed runbook
│
├── data/                                # gitignored; raw + dev sample
├── delta/                               # gitignored; intermediate Parquet/Delta
├── models/                              # serialized PipelineModels + portable exports
│   └── portable/                        # numpy/sklearn artifacts loaded by Streamlit app
│
├── notebooks/                           # one per phase, zero-padded
│   ├── 00_env_check.ipynb
│   ├── 01_ingest.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_classify.ipynb
│   ├── 04_regress.ipynb
│   ├── 05_word2vec.ipynb
│   ├── 06_geo_census.ipynb
│   ├── 08_stream.ipynb
│   ├── 09_lda_trends.ipynb
│   └── 10_bert_embed.ipynb
│
├── src/                                 # reusable Python modules
│   ├── __init__.py
│   ├── config.py                        # paths, constants, seeds
│   ├── spark_setup.py                   # SparkSession bootstrap
│   ├── ingest.py                        # SODA API helpers
│   ├── preprocess.py                    # NLTK + Spark Transformer
│   ├── classify.py                      # classifier training + portable export
│   ├── regress.py                       # regression training + portable export
│   ├── cluster.py                       # Word2Vec + KMeans
│   ├── geo.py                           # spatial join, district fingerprints
│   ├── stream.py                        # Structured Streaming
│   └── lda.py                           # LDA + trend detection
│
├── dashboard/
│   ├── app.py                           # Streamlit entry point
│   ├── tabs/                            # one file per tab for clean separation
│   │   ├── triage_bot.py
│   │   ├── city_pulse.py
│   │   ├── cluster_atlas.py
│   │   ├── bias_audit.py
│   │   ├── live_pulse.py
│   │   └── pipeline_status.py
│   └── assets/                          # static images, NLTK data, GeoJSON
│
└── submission/                          # final zip / report; populated last
```

Notebook naming: `NN_phase-name.ipynb`, zero-padded.

---

## 8. Conventions

- **Python:** type hints on all `src/` functions; docstring with one-line summary + Args/Returns. Notebooks looser but every cell commented.
- **Spark:** `from pyspark.sql import functions as F`; qualify (`F.sum`, `F.row_number`).
- **CSV reads:** historical 2010-19 may need `option("encoding", "ISO-8859-1")`.
- **Sampling:** `sampleBy(col, fractions, seed)` for stratified. Never `randomSplit` for class-sensitive splits.
- **Reproducibility:** all random seeds = `42` unless a specific reason to vary.
- **Dependencies:** track in `requirements.txt`; pin majors. Streamlit Cloud reads this file directly.
- **Comments:** student tone, lowercase first letter is fine, slight imperfections OK.

---

## 9. Known Gotchas

| Gotcha | How to dodge |
|---|---|
| `ModuleNotFoundError: No module named 'src'` inside Spark Python worker when a UDF references our modules | `src/spark_setup.py::get_spark` zips `src/` and calls `sparkContext.addPyFile`. Driver `sys.path.insert` does NOT propagate to worker subprocesses. Discovered Phase 2. |
| Streamlit Cloud has no Java -> can't run PySpark in deployed app | Export models to portable artifacts (numpy / gensim / parquet) at end of each phase |
| Streamlit Cloud 1GB limit | Sub-sample BERT embeddings to ~50K vectors or move to HF Spaces |
| Local Windows + Java 17 PySpark loopback bug | Train on Colab; this avoids the issue entirely |
| 311 column rename in 2020+ | `src/ingest.py` normalizes both schemas |
| Class imbalance (top 10 = >50% of rows) | `sampleBy` stratify + macro-F1 reporting |
| `randomSplit` does not stratify | Use `sampleBy` for train/test splits |
| `nltk.download()` flaky on remote | NLTK data bundled in `dashboard/assets/nltk_data/` |
| SODA API throttles unauthenticated callers at 1000 rows per request | Paginate with `$offset`; consider a free app token if 2M takes too long |
| Em-dashes / smart quotes leak into PDFs | Plain ASCII only in graded text |

---

## 10. Common Commands

```bash
# Local sanity check before pushing
python -m pytest tests/ -q              # if/when tests exist
streamlit run dashboard/app.py          # local dashboard

# Colab: bootstrap a session
!pip install -r requirements.txt -q
!apt-get install -y openjdk-11-jre-headless 2>/dev/null
import os; os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Pull a SODA sample (10K rows for dev)
curl 'https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$limit=10000' > data/sample_10k.csv

# Start an ngrok tunnel from Colab to expose local Streamlit (fallback demo path)
!pip install pyngrok -q
from pyngrok import ngrok
public_url = ngrok.connect(8501)
```

---

## 11. Decisions & Pivots Log

- **2026-05-02** — CLAUDE.md created; scope locked per proposal.
- **2026-05-02** — Compute pivoted from Databricks Free to Google Colab Pro. Reason: shareable-URL demo requirement + user has Pro subscription + 1-2M rows fits local-mode Spark.
- **2026-05-02** — UI pivoted from Plotly Dash to Streamlit. Reason: faster solo-build velocity; better fit for live ML widgets; deploys to Streamlit Community Cloud for free shareable URL.
- **2026-05-02** — Sample size locked at 2M rows stratified, not full 43M. Reason: training iteration speed; still 2x above professor's 1M floor.
- **2026-05-02** — Novelty additions confirmed: Triage Bot centerpiece, BERT embedding sidebar, Bias Audit tab, optional Live Pulse from SODA API.
- **2026-05-02** — Repo created at https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp; initial scaffold pushed (34 files). PAT auth working for Claude-driven git push.
- **2026-05-02** — Phase 0 PASSED on Colab Pro High-RAM. Spark 3.5.8, Java 11, SODA API reachable, normalized schema produces 13 expected columns on 10K sample rows. Pip warning about dataproc-spark-connect/pyspark version mismatch is benign and ignored.
- **2026-05-02** — Phase 1 PASSED. 2M stratified sample on Drive. Discovered label-casing inconsistency: historical labels are ALL-CAPS, 2020+ are Title Case. Same categories, different names. Will collapse via a labels map in Phase 2 preprocessing.
- **2026-05-02** — Phase 2 hit `ModuleNotFoundError: No module named 'src'` on Spark workers (UDF unpickle failure because workers lack project root in sys.path). Fixed by adding addPyFile + executor PYTHONPATH config to `get_spark()`. Worker subprocesses now have `src` importable.
- **2026-05-02** — Phase 2 then hit `LookupError: Resource wordnet not found` on workers (NLTK_DATA_DIR was project-relative, resolved inside the addPyFile zip on workers). Fixed by extending `_NLTK_PATHS` in `src/preprocess.py` to include `/root/nltk_data` and `/usr/share/nltk_data` which are nltk's default search paths.
- **2026-05-02** — Phase 2 PASSED. 1.94M usable rows after empty-token filter. Distinct canonical labels dropped from 244 to 240 via label map. Top tokens domain-specific (loud, music, party, pothole, banging). Note: mean tokens per row = 2.25 (descriptors are short categorical labels rather than free text), so Phase 3 will use 16K HashingTF features instead of 65K.
- **2026-05-02** — Phase 3 PASSED. LR macro-F1 = 0.9593, accuracy = 0.9638. Lifts: +0.95 over majority-class, +0.18 over keyword-heuristic. Training time 40.8 sec on 1.08M rows. 19/20 classes hit F1 ≥ 0.88. Outlier: Noise - Street/Sidewalk F1 = 0.370 (label confusion with Noise - Residential, fixable with location_type as a feature in a future iteration). Portable .npz = 0.10 MB.
- **2026-05-02** — Phase 4 v1 (text + agency + borough + temporal). MAE 113.09h vs baseline 117.52h = +3.8%, below 10% target. Per-category: model wins on high-variance categories (Heat/Hot Water +12.9%, Consumer Complaint +11.0%), is neutral on tight ones. Lesson: text alone has limited signal; the realistic pipeline chains classifier->regressor. Phase 4 v2 retrains with `label_canonical` as a feature to hit target.
- **2026-05-02** — Phase 5 PASSED. Word2Vec vocab 1187 (small because descriptors are short), best k=30 by silhouette 0.526. Synonym probes mixed quality (`rat -> mouse 0.85` good, `rat -> coin` junk). Latent issue discovery delivered: cluster 18 unifies Rodent + Food Establishment + General Construction; cluster 14 unifies Building violations + Dirty Conditions + Consumer Complaint. **Word2Vec quality is the natural motivator for Phase 10 BERT comparison.**

---

## 12. Status

**Current phase:** Phase 5 PASSED (2026-05-02). Word2Vec (vocab 1187, 31 sec) + KMeans k=30 (silhouette 0.526) on 1.32M docs. **Demo highlight: cluster 18 = Rodent 67.3% + Food Establishment 18.0% + General Construction 14.7% — the "urban decay" cluster from the proposal, empirically confirmed.** Other cross-category clusters: cluster 14 (Building/Use + Dirty Conditions + Consumer Complaint), cluster 25 (DOF Property + Street Signs).
**Last touched:** 2026-05-02
**Next action (auto-mode):** user can run any of these in any order: (a) Phase 4 v2 retry cell — 1 min — to hit the 10% MAE target; (b) Phase 6 geographic + census join — 5-10 min; (c) Phase 10 BERT embeddings sidebar (novelty) — 15-20 min on Colab GPU.
**Repo URL:** https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp
**PAT status:** active for this Claude session; expires 2026-06-01 21:06 UTC.
**Phase evidence:** `PRINT.pdf` (Phase 0), `PRINT 2.pdf` (Phase 1), `PRINT 3.pdf` (Phase 2), `PRINT 4.pdf` (Phase 3) — saved locally, gitignored.

**Phase 2-3 lessons baked into the codebase:**
- `src/spark_setup.py::get_spark` zips `src/` and calls `addPyFile` so worker subprocesses can import `src.*` (UDF unpickle was failing without this).
- `src/preprocess.py::_NLTK_PATHS` searches `/root/nltk_data` and `/usr/share/nltk_data` by default so workers find NLTK data via the standard search path.
- `src/classify.py::build_pipeline` defaults to `label_canonical` (post-Phase-2) and 16K hash space (down from 65K — descriptors are short).

**Demo-relevant findings to surface in the report:**
- The keyword-heuristic baseline already hits 0.78 macro-F1, indicating that `problem_detail` is mostly a dictionary lookup (drop-down terms). LR adds discriminative weighting on top to reach 0.96. Frame the achievement honestly in the Q&A.
- Noise - Street/Sidewalk has F1 0.370, getting collapsed into Noise - Residential. Real reporting bias documented by Kontokosta & Hong (2021) — the Bias Audit tab will surface this as a finding.

---

## 13. Pre-Flight Checklist (run mentally before each task)

- [ ] Re-read [`Prompt.txt`](../Prompt.txt). User instruction, very strict.
- [ ] Re-read this CLAUDE.md (it may have changed last session).
- [ ] Skim §11 Decisions log for any pivots since last session.
- [ ] If writing prose for the prof: apply §2.4 anti-AI-detection rules.
- [ ] If writing code: every block needs a student-tone comment.
- [ ] If a major decision is needed: ask the user; offer a recommendation.
