# Step-by-Step Implementation Guide

> Micro-detailed runbook for the Big Data final project.
> One section per phase. Every command is copy-paste ready.
> Update this file when reality diverges from the plan.

**Last updated:** 2026-05-02
**Working directory:** `FINAL PROJECT IMPLEMENTATION/`

---

## How to use this guide

Each phase is laid out as:

1. **Goal** — what we're trying to produce
2. **Pre-flight** — what to read or check first
3. **Steps** — numbered, copy-paste-ready commands
4. **Expected output** — what success looks like
5. **If it fails** — fall-back paths
6. **Done when** — the gate before moving on

Always re-read [`../Prompt.txt`](../Prompt.txt) before starting a phase. User instruction, very strict.

---

## Phase 0 — Environment Lock-In

### Goal
A Colab Pro notebook that boots PySpark in local mode, mounts Drive, reads 10,000 rows from the NYC 311 SODA API, and prints the schema.

### Pre-flight

- Confirm Colab Pro subscription is active (https://colab.research.google.com -> top-right account menu).
- Have the GitHub repo created (Step 0.A below) so notebooks can be pushed there from Colab.

### Steps

**0.A — Create the GitHub repo (one-time)**

1. Go to https://github.com/new while logged in as your GitHub account.
2. Repo name: `cs-gy-6513-big-data-311-nlp`.
3. Visibility: **Public** (required for Streamlit Community Cloud free tier).
4. Initialize: tick "Add a README file", select "Python" .gitignore, MIT or no license.
5. Click "Create repository".
6. Copy the repo URL.

**0.B — Push the local scaffold to GitHub (one-time)**

From a terminal (Git Bash or PowerShell) inside `FINAL PROJECT IMPLEMENTATION/`:

```bash
git init
git add CLAUDE.md README.md requirements.txt .gitignore docs/ src/ notebooks/ dashboard/
git commit -m "Initial scaffold for 311 NLP project"
git branch -M main
git remote add origin https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp.git
git push -u origin main
```

**0.C — Open the notebook in Colab**

1. Go to https://colab.research.google.com.
2. File -> Open notebook -> GitHub tab.
3. Paste your repo URL. Pick `notebooks/00_env_check.ipynb`.
4. Top-right: switch runtime to **GPU (T4)** via Runtime -> Change runtime type. Not strictly needed for Phase 0 but warms it up for Phase 10.

**0.D — Run the notebook cell by cell**

The notebook does the following. Each cell is also runnable standalone:

1. **Cell 1** mounts Google Drive at `/content/drive` for persistent storage.
2. **Cell 2** clones the GitHub repo into `/content/project` so we can `import src.*`.
3. **Cell 3** installs pinned dependencies from `requirements.txt`.
4. **Cell 4** installs Java 11 (PySpark prefers Java 11 over the Java 17 that ships with Colab) and sets `JAVA_HOME`.
5. **Cell 5** boots a `SparkSession` with Delta Lake extensions.
6. **Cell 6** hits the SODA API for 10,000 rows from the 2020+ dataset, prints the first 5 rows, and the schema.

### Expected output

- Cell 5 prints: `Spark version: 3.5.x`, `Java version: 11.x.x`.
- Cell 6 prints a Spark DataFrame schema with ~40 columns including `unique_key`, `created_date`, `complaint_type`, `descriptor`, `borough`.

### If it fails

- **`Java not found`:** rerun Cell 4. The `apt-get` install can be flaky on first session.
- **SODA API timeout:** SODA throttles unauthenticated callers. Get a free app token at https://data.cityofnewyork.us/profile/edit/developer_settings and add it as a Colab secret (left-side key icon) named `SODA_APP_TOKEN`.
- **Drive mount fails:** Colab sometimes asks for re-auth. Click the popup and allow.
- **`pyspark` import error after install:** Restart runtime (Runtime -> Restart runtime) and rerun from Cell 4.

### Done when

`notebooks/00_env_check.ipynb` runs top-to-bottom on a fresh Colab kernel and the schema is printed. **At this point, save the notebook back to GitHub via File -> Save a copy in GitHub.**

---

## Phase 1 — Ingest + Normalize

### Goal
A 2,000,000-row stratified sample of NYC 311 complaints, written to Parquet on Drive, with column names normalized across the 2020+ and 2010-19 schemas.

### Pre-flight

- Phase 0 done.
- ~30-60 min of Colab runtime budget. SODA API pagination at 50K rows per request means ~40 requests for 2M rows.

### Steps

**1.A — Open `notebooks/01_ingest.ipynb`** (it imports from `src.ingest`).

**1.B — Run cells in order:**

1. **Cell 1** loads the SparkSession from `src.spark_setup`.
2. **Cell 2** calls `src.ingest.fetch_311_to_parquet(start_year=2020, target_rows=1_000_000, out_path="/content/drive/MyDrive/cs6513/raw/2020plus")`.
3. **Cell 3** calls the same for 2010-2019, target 1M rows.
4. **Cell 4** loads both Parquets, normalizes column names via `src.ingest.normalize_columns`, unions them.
5. **Cell 5** stratifies via `sampleBy` on `complaint_type` to exactly 2M rows, writes to `/content/drive/MyDrive/cs6513/sample_2m.parquet`.
6. **Cell 6** prints class distribution histogram (top 20 categories).

### Expected output

- Cell 5: `Sample written. df.count() = 2000000`.
- Cell 6: top 5 categories typically `Noise - Residential`, `HEAT/HOT WATER`, `Illegal Parking`, `Blocked Driveway`, `Street Condition`.

### If it fails

- **Out of memory:** Colab Pro standard runtime has 12.7GB RAM; if you blew it, switch to "High-RAM" runtime (Runtime -> Change runtime type -> "High-RAM").
- **SODA returns fewer rows than expected:** the historical dataset cap may have moved. Inspect `df.count()` after each pull and adjust `target_rows` upward.
- **Column normalization missing a field:** the normalization map lives in `src/ingest.py::COLUMN_MAP`. Add the new field there and rerun Cell 4.

### Done when

`/content/drive/MyDrive/cs6513/sample_2m.parquet` exists, `df.count() == 2_000_000`, top-20 categories histogram saved to `dashboard/assets/class_dist.png`.

---

## Phase 2 — Text Preprocessing

### Goal
A reusable Spark Transformer that tokenizes, lowercases, removes stopwords, and lemmatizes the `descriptor` field.

### Steps

1. Run `notebooks/02_preprocess.ipynb`.
2. Cell 1: download NLTK data into `dashboard/assets/nltk_data/` (one-time; commit to repo).
3. Cell 2: instantiate `src.preprocess.TextPreprocessor`.
4. Cell 3: apply to the 2M sample, write to `delta/sample_2m_preprocessed.parquet`.

### Done when

Preprocessing finishes in <2 min on the 2M sample, output `tokens` column non-null in the 99th percentile.

---

## Phase 3 — Classification

### Goal
Spark MLlib classifier with macro-F1 >= 0.75 on top 20 categories, plus a portable export the Streamlit app can load without Spark.

### Steps

1. Run `notebooks/03_classify.ipynb`.
2. Build TF-IDF features (`HashingTF` with 65536 features + `IDF`).
3. Stratified split via `src.classify.stratified_split` (uses `sampleBy`).
4. Fit Logistic Regression (multinomial) and Random Forest.
5. Evaluate on test set: macro-F1, per-class precision/recall, accuracy.
6. Compare against majority-class baseline + keyword baseline.
7. Save best model to `models/classifier_pipeline/`.
8. Export portable artifact: `src.classify.export_portable(model, "models/portable/classifier.npz")`.

### Done when

- Macro-F1 >= 0.75 on top 20 categories.
- Confusion matrix saved to `dashboard/assets/cm.png`.
- Portable `.npz` (vocab + IDF + LR coefficients) loads in pure-Python and produces identical predictions.

---

## Phase 4 — Resolution Time Regression

### Goal
A regressor predicting hours-to-resolve from text + structured features, beating median-per-category baseline by >=10% MAE.

### Steps

1. Run `notebooks/04_regress.ipynb`.
2. Compute label `resolution_hours = (closed_date - created_date) / 3600`. Drop rows where this is null or negative.
3. Features: TF-IDF (capped at 4096), Agency one-hot, Borough one-hot, hour-of-day, day-of-week.
4. Fit Linear Regression and Random Forest Regressor.
5. Evaluate MAE, RMSE, R^2 on held-out 20%.
6. Save portable artifact.

### Done when

- Test MAE >= 10% improvement over median baseline.
- Feature importance plot saved.

---

## Phase 5 — Word2Vec + Clustering

### Goal
Word2Vec embeddings + K-Means clusters with positive silhouette, top-10 terms per cluster human-readable.

### Steps

1. Run `notebooks/05_word2vec.ipynb`.
2. Train `pyspark.ml.feature.Word2Vec` (vector size 100, window 5, minCount 5).
3. Aggregate doc embeddings via averaging.
4. Sweep K-Means k in [5, 30]; pick by silhouette.
5. For each cluster, surface top-10 most-frequent terms.
6. Export Word2Vec as gensim `KeyedVectors` for use in the Streamlit app.

### Done when

- Silhouette curve plot saved.
- Top-10 terms per cluster visible and "make sense" (rats / trash / vacant lot grouped, etc.).

---

## Phase 6 — Geographic + Census Join

### Goal
Per-district complaint vocabulary fingerprint table that surfaces district-distinctive terms.

### Steps

1. Run `notebooks/06_geo_census.ipynb`.
2. Load community districts GeoJSON (`mzpm-a6vd`) and ACS demographics.
3. Spatial join 311 lat/long to district polygons using Shapely / GeoPandas (run on driver, not Spark, since it's only 71 polygons).
4. Compute TF-IDF per district vs. corpus average; top-10 distinctive terms per district.
5. Save as Parquet for the Cluster Atlas tab.

### Done when

`delta/district_fingerprints.parquet` exists; per-district top terms table reviewable.

---

## Phase 7 — Streamlit Dashboard (central deliverable)

### Goal
A live Streamlit app at a public URL with five working tabs.

### Steps

1. Confirm `models/portable/` has all required artifacts: `classifier.npz`, `regressor.npz`, `word2vec.kv`, `district_fingerprints.parquet`.
2. Implement each tab in `dashboard/tabs/*.py`:
   - `triage_bot.py`: text input -> classify (numpy LR) -> regress -> nearest cluster -> top-3 historical neighbors -> Folium map pin
   - `city_pulse.py`: choropleth + time series
   - `cluster_atlas.py`: cluster list + per-cluster top terms + per-district word clouds
   - `bias_audit.py`: per-borough macro-F1 chart, predicted vs actual resolution scatter
   - `pipeline_status.py`: dataset stats, model versions, training metadata
3. Run locally: `streamlit run dashboard/app.py`. Verify all tabs render.
4. Commit + push to GitHub.
5. Deploy to Streamlit Community Cloud:
   - Go to https://streamlit.io/cloud (sign in with GitHub)
   - Click "New app"
   - Pick your repo, branch `main`, main file `dashboard/app.py`
   - Click Deploy
6. App URL is shareable. Test on a phone / from another browser.

### Done when

- Public URL works.
- Triage Bot accepts a complaint and returns a classification + resolution + cluster + map in <2 sec.
- Demo flow rehearsed in <3 min.

---

## Phase 8 — Streaming PoC

### Goal
A Spark Structured Streaming job that watches a folder, classifies new rows, and writes to Delta. The dashboard's Live Pulse tab tails this Delta sink.

### Steps

1. Run `notebooks/08_stream.ipynb`.
2. Configure stream: `spark.readStream.option("maxFilesPerTrigger", 1).csv("watch_folder/")`
3. Apply trained `PipelineModel` via `model.transform`.
4. Sink: `writeStream.format("delta").option("checkpointLocation", ...).trigger(Trigger.AvailableNow).start("delta/streaming_out/")`
5. In Streamlit, the Live Pulse tab reads `delta/streaming_out/` periodically (polling every 5 sec).

### Done when

Dropping a CSV into `watch_folder/` produces classified output visible in the dashboard within one micro-batch.

---

## Phase 9 — LDA + Trend Detection

### Goal
LDA topics that surface themes beyond the existing taxonomy + a trend detector flagging recent term-frequency spikes.

### Steps

1. Run `notebooks/09_lda_trends.ipynb`.
2. Fit `pyspark.ml.clustering.LDA` with k=20-30.
3. For each topic, get top 10 terms; manually label them.
4. Sliding window: compute term-frequency over 30-day windows; z-score each term's recent vs historical frequency; surface terms with z > 3.
5. Save labeled topics + flagged spikes for the City Pulse tab.

### Done when

At least one interpretable topic + one flagged spike with date range documented.

---

## Phase 10 — BERT Embedding Sidebar (novelty)

### Goal
Side-by-side comparison of Word2Vec clusters vs sentence-transformer clusters in the Cluster Atlas tab.

### Steps

1. Run `notebooks/10_bert_embed.ipynb`.
2. Load `sentence-transformers/all-MiniLM-L6-v2` on Colab GPU.
3. Encode a stratified sub-sample (50K-100K rows is plenty for clustering, keeps the parquet under 200MB).
4. Run K-Means on BERT embeddings with the same k chosen in Phase 5.
5. Compute silhouette for both methods; produce a comparison plot.
6. Save embedded sub-sample + cluster labels to `delta/bert_clusters.parquet`.

### Done when

Comparison chart shows silhouette difference; Cluster Atlas tab has a toggle "Word2Vec | BERT".

---

## Phase 11 — Submission Packaging *(deferred)*

Held until user signal. Will produce: README polish, repo cleanup, demo rehearsal script, report PDF.

---

## Daily / Per-Session Workflow

```
1. Open Colab Pro
2. Open the notebook for the current phase from GitHub
3. Mount Drive (Cell 1)
4. Run the bootstrap cell that pulls latest from GitHub:
   !cd /content/project && git pull
5. Work in cells; commit small changes to GitHub via File -> Save a copy in GitHub
6. End of session: update CLAUDE.md §11 (Decisions Log) and §12 (Status)
7. End of phase: tick the "Done when" gate; check in here for the next phase
```

## Demo Day Checklist

- [ ] Streamlit URL is live and accessible from a fresh browser
- [ ] All five tabs render correctly
- [ ] Triage Bot returns results in <2 sec for a typed complaint
- [ ] Demo script under 12 min so we have buffer for Q&A
- [ ] Backup ngrok URL ready (in case Streamlit Cloud is down)
- [ ] All teammates have the URL and have practiced the click-through
- [ ] Slides cover: 1 slide context, 1 slide architecture, 10 min demo, 1 slide future work, 1 slide thanks
