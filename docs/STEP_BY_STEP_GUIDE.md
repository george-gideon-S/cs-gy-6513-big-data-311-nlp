# Step-by-Step Implementation Guide

**Last updated:** 2026-05-03
**Project:** CSGY-6513 Big Data final project, NYC 311 NLP pipeline.
**Working directory on Windows:** `G:\MASTER'S\NYU SEMESTER 02\Big Data\FINAL PROJECT\FINAL PROJECT IMPLEMENTATION\`
**Repo state at time of writing:** main branch tip at commit `24e9c1d` (dashboard
button visibility, sample-size string fixes 2M to 10M, notebook ref consolidation,
gitignore re-allowing runtime artifacts so the deployed dashboard ships with data
in the repo). Earlier relevant commits: `fe3ee6b` (commit_artifacts no-ops on
gitignored paths), `8effe34` (SODA pull is now crash-safe with per-page parquet
checkpoints and an 8-retry budget with up to 60s backoff), `8907645` (this guide
and `docs/REPORT.tex` were rewritten), `b9d74d6` (the repo cleanup that collapsed
the eight phase notebooks into the single submission notebook).

---

## What this guide is for

This guide walks you through running the entire end-to-end pipeline from a clean
Colab Pro environment. The repo used to have eight separate phase notebooks. After
the recent cleanup we collapsed everything into a single notebook,
`notebooks/FINAL_PROJECT_311_NLP.ipynb`, which runs every phase top to bottom.
The notebook handles ingest, preprocessing, classifier training, regressor
training, Word2Vec, borough fingerprints, BERT MiniLM, and final artifact export.
Use this guide if you are running the project for the first time, recovering from a
crash, or trying to deploy the dashboard to Streamlit Cloud.

---

## Prerequisites

Before you start, make sure you have:

1. **Colab Pro account** with at least the standard subscription. The free tier will
   not survive the 10M ingest. You need High-RAM and either an H100 or A100 to fit
   the BERT phase comfortably.
2. **GitHub repo access.** The repo lives at
   `https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp` and is public.
   You need write access if you plan to commit artifacts back from Colab.
3. **A SODA app token** from NYC OpenData. Get one for free at
   `https://data.cityofnewyork.us/profile/edit/developer_settings`. Without a token
   the SODA endpoints will throttle you to a few hundred rows per minute, which
   makes the 10M sample impossible. Add it to Colab Secrets as `SODA_APP_TOKEN`.
4. **A GitHub Personal Access Token (fine-grained).** Add it to Colab Secrets as
   `GITHUB_PAT` so the auto-commit cells can push artifacts back to the repo.
5. **Google Drive with at least 4 GB free** mounted into Colab. The intermediate
   parquet files plus the final artifacts add up to roughly 2.5 GB.
6. (Optional) **A Streamlit Community Cloud account** signed in via GitHub. Free
   tier works as long as the deployed bundle stays under 1 GB. The portable
   artifacts we ship are sized to fit.

---

## The single end-to-end runbook

The notebook is one file with cells grouped into clearly labeled phases. You can
either run it with **Run All** or step through phases one at a time. Each phase
has a resume gate so re-running the notebook will skip phases whose final
artifacts already exist on Drive.

### Step 1: Open the notebook in Colab

1. Go to `https://colab.research.google.com`.
2. File -> Open notebook -> GitHub tab.
3. Paste `https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp` into the
   search box. Pick `notebooks/FINAL_PROJECT_311_NLP.ipynb`.
4. The notebook opens with all output cleared if you opened it fresh from GitHub.

### Step 2: Pick the runtime

1. Runtime -> Change runtime type.
2. Hardware accelerator: **GPU**, GPU type **A100** if available, **H100** if your
   tier offers it. The T4 free-tier GPU technically works for Phase 10 but encoding
   100K MiniLM embeddings drops from ~3 minutes on A100 to ~25 minutes on T4.
3. Runtime shape: **High-RAM**. Standard 12.7 GB will OOM during the 10M Spark
   shuffle in Phase 1.
4. Click Save. Colab will reconnect with the new runtime.

### Step 3: Run-all behavior and what to expect

Hit Runtime -> Run all. The notebook does the following in order. Approximate
runtimes are for an A100 + High-RAM Colab Pro session.

| Phase | What it does | Approx. runtime |
|:-----:|:-------------|:----------------|
| 0 | Bootstrap: detect Colab, mount Drive, clone repo to /content/project, install pinned deps from requirements.txt, install Java 11, set JAVA_HOME, boot Spark in local[*] mode | 4-6 min on first session, ~30 sec if deps cached |
| 1 | Ingest: pull 5M from erm2-nwe9 + 5M from 76ig-c548 via SODA paginated requests, write a `.raw.parquet` checkpoint via pyarrow, then convert to Spark DataFrame and write `sample_2m.parquet` | 8-12 min for the SODA pull, 2-3 min for the Spark conversion |
| 2 | Preprocess: NLTK tokenize, lowercase, stopwords, lemmatize on the descriptor field. Write `sample_2m_preprocessed.parquet` | 3-5 min |
| 3 | Classifier: TF-IDF (HashingTF 16K + IDF), stratified 80/20 split via sampleBy, fit multinomial Logistic Regression. Evaluate macro-F1 vs majority-class and keyword baselines. Save Spark pipeline + portable .npz | 6-9 min |
| 4 | Regressor: log1p(resolution_hours) target. v1 = TF-IDF 4K + agency + borough + hour-of-day + day-of-week. v2 = same plus label_canonical one-hot. Fit Linear Regression both. Evaluate MAE vs median-per-category baseline. Save portable .npz | 5-7 min |
| 5 | Word2Vec: vector size 100, window 5, minCount 5. KMeans sweep k=5..30, pick by silhouette. Cross-tab cluster vs official category. Save .kv (gensim KeyedVectors) | 8-12 min |
| 6 | Borough fingerprints: per-borough TF-IDF lift = freq_in_borough / freq_in_corpus. Filter to terms with >=100 occurrences. Save fingerprints parquet | 2-3 min |
| 10 | BERT MiniLM: load all-MiniLM-L6-v2, encode 100K stratified subsample to 384-dim embeddings, KMeans sweep, silhouette comparison vs Phase 5. Save bert_clusters.parquet | 4-6 min on A100, 25+ min on T4 |
| Final | Auto-commit cell: stage portable artifacts under `models/portable/` and `dashboard/assets/`, push to GitHub via PAT-embedded URL | ~30 sec |

Total wall-clock: roughly **45-60 minutes on a fresh A100 session**.

### Step 4: Where artifacts land

Everything that needs to survive across sessions goes to Drive. Everything the
Streamlit dashboard reads at request time gets committed back to the GitHub repo
under `dashboard/assets/`.

```
/content/drive/MyDrive/cs6513/
  raw/
    2020plus.parquet/         # SODA pull from erm2-nwe9
    2010_2019.parquet/        # SODA pull from 76ig-c548
    .raw.parquet              # pyarrow checkpoint, written before Spark conversion
  sample_2m.parquet/          # 10M-row stratified sample (yes, file is named "2m" -- legacy)
  sample_2m_preprocessed.parquet/   # post-NLTK
  models/
    classifier_pipeline/      # Spark MLlib pipeline, only readable in Spark
    regressor_v1_pipeline/
    regressor_v2_pipeline/
    word2vec.model/           # Spark Word2Vec model
  delta/
    bert_clusters.parquet/
    district_fingerprints.parquet/

/content/project/         (repo clone)
  models/portable/
    classifier.npz          # vocab + IDF + LR coefficients, pure-numpy loadable
    regressor_v1.npz
    regressor_v2.npz
    word2vec.kv             # gensim KeyedVectors export
  dashboard/assets/
    cm.png                  # confusion matrix
    silhouette_curve.png
    bert_vs_w2v.png
    class_dist.png
    nyc_boroughs.geojson    # already in repo
    metrics.json            # headline numbers, dashboard reads at startup
```

### Step 5: What to do on crash

The notebook is built to be resumable. If a phase crashes you do not have to redo
the earlier phases.

1. **JVM crash (Phase 1, 2, 3, 4, 5).** Symptoms are `Py4JJavaError` or
   `Py4JNetworkError: An error occurred while trying to connect to the Java server`.
   Run the cell labeled "Reset Spark session" near the top of the bootstrap. It
   calls `src.spark_setup.reset_spark()` which clears stale JVM singletons and
   reboots a fresh `SparkSession`.
2. **OOM during Spark shuffle.** Switch to High-RAM runtime if you have not
   already. If still OOMing, drop the sample size by editing the
   `TARGET_ROWS_PER_ENDPOINT` constant in the Phase 1 config cell from 5_000_000 to
   3_000_000.
3. **SODA timeout mid-pull.** The `.raw.parquet` checkpoint preserves whatever
   pages you already pulled. Re-run the Phase 1 cell. The ingest function checks
   for the checkpoint first and resumes from where it stopped.
4. **Drive disconnect.** Colab sometimes times out the Drive mount on long
   sessions. Re-run the Drive mount cell (it is the second cell in the bootstrap)
   and continue. Drive auto-syncs anything you wrote during the disconnect.
5. **Disk full on /content.** Colab gives you ~100 GB on the local SSD. The raw
   parquet checkpoints can take 5-6 GB temporarily. If you fill it up, the
   `del df_raw_pandas` cell after the Spark conversion frees most of it. If that
   does not help, restart the runtime and resume from the next phase.

---

## If something goes wrong

Read these before you spend an hour on Stack Overflow. These are the failure
modes we have actually hit during runs.

### JVM crashes mid-phase (Spark dies)

The most common cause is two SparkSessions trying to bind to the same port.
Symptoms in the cell output:

```
Py4JNetworkError: An error occurred while trying to connect to the Java server
java.net.BindException: Address already in use
```

Fix: run the reset cell.

```python
from src.spark_setup import reset_spark, get_spark
reset_spark()
spark = get_spark()
```

`reset_spark` in `src/spark_setup.py` calls `SparkSession.builder.getOrCreate().stop()`
and clears `SparkContext._active_spark_context` so the next `get_spark()` returns a
brand new session. If that does not work, restart the Colab runtime
(Runtime -> Restart runtime). Drive stays mounted across restarts most of the time.

### Partial parquet directory

A Spark write that gets interrupted leaves a `_temporary` directory next to the
parquet directory. If you try to read it back, Spark errors with
`Unable to infer schema for Parquet`. Delete the partial directory and re-run the
write cell.

```python
import shutil, os
target = "/content/drive/MyDrive/cs6513/sample_2m.parquet"
if os.path.exists(target + "/_temporary"):
    shutil.rmtree(target)
```

The Phase 1 ingest checkpoint at `.raw.parquet` is a single pyarrow file rather
than a Spark directory, so it does not have this problem.

### Pip dep conflicts

Colab pre-installs a lot of stuff and sometimes a pinned dep clashes. Symptom is
a `requirements.txt` install that finishes but then an `import` later fails with
`undefined symbol` or `attribute X not found`. Fix:

```bash
!pip install --force-reinstall --no-deps gensim==4.3.2 numpy==1.26.4
```

The two libraries that consistently fight are `gensim` and `numpy`. We pin
`numpy>=1.24` (no upper bound) because both Colab and Streamlit Cloud ship
numpy 2.x. If gensim breaks against numpy 2, use `--force-reinstall --no-deps`
to keep gensim and not let it pull a downgrade.

### OOM (out of memory)

Three flavors:

1. **OOM in Spark driver.** Lower `spark.driver.memory` is wrong here. You want it
   higher. In `src/spark_setup.py::get_spark`, the default is
   `spark.driver.memory=12g`. Bump it to `16g` if you are on High-RAM.
2. **OOM in pandas conversion.** This happens when we do `.toPandas()` on the
   full 10M sample for plotting. We never do that on purpose, but if a notebook
   cell tries to materialize too many rows it will die. Sample first:
   `df.sample(False, 0.001).toPandas()`.
3. **OOM during BERT encoding.** MiniLM at batch_size=128 fits comfortably on
   A100. On T4 you have to drop to batch_size=32 and the wall-clock balloons.

### GPU not detected

Symptom: Phase 10 runs but it says `device='cpu'` in the encode loop. Fix:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

If `cuda.is_available()` is False, you forgot to pick a GPU runtime. Runtime ->
Change runtime type -> GPU. If it is True but the encoder still uses CPU, that is
a sentence-transformers default. Force it explicitly:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
```

### SODA token rejected

Symptom: 403 from the SODA API after ~10 requests. Means your token is wrong or
expired. Get a new one from `https://data.cityofnewyork.us/profile/edit/developer_settings`,
update the Colab Secret `SODA_APP_TOKEN`, and re-run Phase 1.

### Recovering from a SODA timeout mid-pull

The two SODA endpoints occasionally drop the connection mid-page or return a 504
when you push past a couple million rows in a single pull. Symptoms include
`requests.exceptions.ReadTimeout`, `ConnectionResetError`, or a hung cell that
never finishes its progress bar. This used to mean restarting the entire SODA
pull from page zero, which on a 10M run was an hour of wasted compute.

After commit `8effe34` the ingest function is crash-safe via per-page parquet
checkpoints. Each page (50K rows by default) is written immediately to disk under
`<endpoint>.partial/part_NNNNN.parquet` before the next page is fetched, where
`NNNNN` is the page index zero-padded to five digits. The function also wraps
each page request in an 8-retry budget with exponential backoff up to 60
seconds, so transient blips no longer abort the run.

The user-side recovery procedure is now: **just re-run the Phase 1 cell.** When
the function starts, it scans `<endpoint>.partial/` for existing
`part_NNNNN.parquet` files, infers the highest already-written page index, and
resumes the SODA call from the next offset. Pages that are already on disk are
skipped entirely. There is nothing to delete and no offset to set manually.
Concretely the layout looks like:

```
/content/drive/MyDrive/cs6513/raw/
  2020plus.partial/
    part_00000.parquet     # rows 0..49999
    part_00001.parquet     # rows 50000..99999
    ...
    part_00099.parquet     # last good page before timeout
  2010_2019.partial/
    part_00000.parquet
    ...
```

Once the resumed run reaches the configured `TARGET_ROWS_PER_ENDPOINT`, the
function concatenates the parts into a single Parquet directory at
`<endpoint>.parquet/` and the `.partial/` directory is left in place as a
recovery artifact. You can delete the `.partial/` directory manually after a
clean Phase 1 finishes, but it does not hurt to leave it; Phase 2 reads only
from the consolidated `<endpoint>.parquet/` path.

If the resumed run still times out mid-page, the retry budget should kick in
before you see an exception. If you do see an exception, just re-run the cell
again. The state on disk is monotonic: every successful re-run only adds parts,
it never deletes them.

### Streamlit Cloud build fails

Most common cause is a binary dep that does not have a wheel. We have already
filtered `requirements.txt` to wheel-only deps; the file `requirements-train.txt`
is for Colab training only and includes pyspark, which Streamlit cannot install
on its 1 GB free tier. Make sure the deployed app reads from `requirements.txt`
not `requirements-train.txt`.

---

## How to re-run only one phase

The notebook uses a resume-gate pattern: at the top of every phase there is a
check like

```python
final_artifact = "/content/drive/MyDrive/cs6513/sample_2m_preprocessed.parquet"
if os.path.exists(final_artifact):
    print(f"Phase 2 already complete -- skipping. Delete {final_artifact} to re-run.")
    df_pre = spark.read.parquet(final_artifact)
else:
    # ... run the phase ...
```

So to re-run only Phase 4 (the regressor):

1. Delete the Phase 4 final artifact:
   ```python
   import shutil
   shutil.rmtree("/content/drive/MyDrive/cs6513/models/regressor_v2_pipeline", ignore_errors=True)
   shutil.rmtree("/content/drive/MyDrive/cs6513/models/regressor_v1_pipeline", ignore_errors=True)
   ```
2. Run all cells from the top. Phases 0, 1, 2, 3 will all hit their gates and
   skip in a couple of seconds each. Phase 4 will rerun fresh.
3. Phase 5 onward may also re-run if Phase 4's artifacts are an input. Check the
   gate cell at the top of each downstream phase.

If you want to re-run from a specific cell forward without doing Run All, use
Runtime -> Run after to start from your current cell and continue to the end.

---

## Submitting the artifacts to GitHub

Every phase ends with an auto-commit cell that pushes the portable artifacts back
to the repo so the deployed Streamlit app can read them at startup. The pattern
is in `src/colab_git.py`.

### One-time GITHUB_PAT setup

1. Go to `https://github.com/settings/personal-access-tokens/new`.
2. Pick **Fine-grained**. Resource owner = your account. Repo access =
   `cs-gy-6513-big-data-311-nlp` only.
3. Permissions: under Repository permissions, set **Contents = Read and write**.
   That is the only one you need.
4. Pick an expiry date that covers the project window.
5. Generate. Copy the token immediately (you cannot see it again).
6. In Colab, click the key icon on the left sidebar (Secrets). Add a new secret:
   name `GITHUB_PAT`, value the token you just copied. Toggle Notebook access on.

### What commit_artifacts does

The function lives in `src/colab_git.py`. From inside the notebook:

```python
from src.colab_git import commit_artifacts
commit_artifacts(
    paths=["dashboard/assets/cm.png", "dashboard/assets/metrics.json", "models/portable/classifier.npz"],
    message="phase 3: classifier macro-F1 0.959, portable npz exported"
)
```

Internally it:

1. Reads `GITHUB_PAT` from `google.colab.userdata.get("GITHUB_PAT")`.
2. Configures `git config user.email` and `git config user.name` because Colab
   has neither set by default.
3. Stages exactly the paths you passed. We never use `git add .` on Colab because
   notebook output cells contain large blobs we do not want in the repo.
4. Commits with the supplied message.
5. Pushes via a one-shot URL of the form
   `https://x-access-token:$TOKEN@github.com/...` so the token never lands in
   `.git/config` on the Colab disk.

If you accidentally delete the Colab secret mid-session, the function will print
a helpful error pointing you back here.

### What gets committed back, and why it matters for the dashboard

After commit `24e9c1d` the repo `.gitignore` has been relaxed so that the small
runtime artifacts the dashboard reads at startup are tracked under version
control. Specifically the JSON metric blobs, the gensim `KeyedVectors` `.kv`
file, and the numpy `.npz` model exports under `models/portable/` and
`dashboard/assets/` are committed; the heavy parquet directories, the Spark
PipelineModels, and any per-row intermediate files remain gitignored. The
practical consequence is that as soon as the auto-commit cell at the end of the
notebook lands, the deployed Streamlit dashboard pulls the new artifacts on its
next rebuild and the live demo starts serving the most recent run's numbers,
without any manual upload step.

If a path you pass to `commit_artifacts` happens to be gitignored, the function
no longer aborts the entire commit (this was the change in commit `fe3ee6b`).
Instead it silently skips the gitignored entry, logs which paths were skipped,
and proceeds to commit whatever remains. This means you can call the function
with the same list of paths whether or not your gitignore is in the relaxed or
strict state; the behavior degrades gracefully.

### A note on the dashboard buttons

The dashboard's primary action buttons used to render with white text on a
slightly off-white background, which on some screens made the labels nearly
invisible. Commit `24e9c1d` fixed this by overriding the Streamlit theme so the
buttons now show **white text on a purple background** matching the rest of the
dashboard's primary accent color. This is purely a CSS fix; if you fork the
dashboard and the buttons revert, check `dashboard/app.py` for the `st.markdown`
block that injects the override styles and confirm it is still wired in before
the first `st.button` call.

---

## Streamlit Cloud deployment

### One-time setup

1. Sign in to `https://streamlit.io/cloud` with your GitHub account.
2. Click "New app".
3. Repository: `george-gideon-S/cs-gy-6513-big-data-311-nlp`.
4. Branch: `main`.
5. Main file path: `dashboard/app.py`.
6. Python version: pick the version pinned in `runtime.txt` (currently 3.11).
7. Advanced settings: leave defaults.
8. Click Deploy. First build takes 6-10 minutes because it has to install
   numpy + gensim + streamlit + pydeck + folium fresh.

The app URL takes the form `https://nyc-311-triage.streamlit.app/` (the exact
slug depends on the name you picked when you clicked New app).

### Auto-redeploy on push

Streamlit Cloud watches the `main` branch and rebuilds the app on every push.
That means as soon as the auto-commit cell at the end of the notebook lands, the
deployed app gets the new artifacts within ~3-5 minutes. There is no separate
deploy step. If a build fails, the Streamlit dashboard sidebar shows you the
build log; the most common cause is a dep added to `requirements-train.txt`
instead of `requirements.txt`.

### Local dashboard test before pushing

Before you push artifacts that will trigger a rebuild, sanity check the
dashboard locally:

```bash
cd "G:/MASTER'S/NYU SEMESTER 02/Big Data/FINAL PROJECT/FINAL PROJECT IMPLEMENTATION"
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The local app will load the portable artifacts from `models/portable/` and
`dashboard/assets/` exactly the way the deployed one does. If it crashes locally,
it will crash on Streamlit Cloud too.

---

## Verifying success

After Run All finishes, check these numbers in the cell outputs. The dataset
is the **10M-row** stratified sample (5M from each SODA endpoint, drawn from a
roughly 43M-row 13 GB raw corpus). The "healthy" ranges below are pinned to a
recent successful run; absolute numbers from a fresh 10M run will move slightly
because of stratification randomness, but the order of magnitude and the
relative ordering between baselines and models should match. If your numbers
are dramatically outside this range, something went wrong.

### Phase 1 (ingest)
- 2020+ pull: roughly 5,000,000 rows from erm2-nwe9
- 2010-19 pull: roughly 5,000,000 rows from 76ig-c548
- Combined sample: 10,000,000 rows total, written to `sample_2m.parquet`
- Top 5 categories visible in the histogram cell. Common top entries:
  Noise - Residential, HEAT/HOT WATER, Illegal Parking, Blocked Driveway, Street Condition

### Phase 2 (preprocess)
- Output rows: roughly 1.94M usable after empty-token filter (the rest are
  rows whose descriptor field had no surviving tokens after stopword and
  lemma cleanup)
- Mean tokens per row: ~2.25 (descriptors are short categorical labels, not
  free-form text)

### Phase 3 (classifier)
- Macro-F1 on the holdout: target was 0.75, expected ~0.96 on 2M, similar on 10M
- Lift over keyword baseline: ~0.18 (keyword baseline alone hits ~0.78 because
  descriptors basically encode the label)
- Confusion matrix saved to `dashboard/assets/cm.png`
- The Noise-Street/Sidewalk class is the worst per-class F1 at ~0.37 (it
  collapses into Noise-Residential due to overlapping descriptor vocabulary)

### Phase 4 (regressor)
- v1 lift over median-per-category baseline: ~3.9% MAE
- v2 lift: ~4.1% MAE
- Both below the 10% target overall, but per-category lift hits 10% on
  Heat/Hot Water and other high-variance categories
- Low-variance categories (Bulky Item, Police Matter) are SLA-bounded so text
  features add no signal

### Phase 5 (Word2Vec + KMeans)
- Vocab size: ~1,187 words after the minCount=5 filter
- Best k: 25, silhouette ~0.4635
- Top-10 terms per cluster should look thematic (rats/trash/vacant lot
  cluster, vehicle abandoned cluster, heat/hot water cluster, etc.)

### Phase 6 (borough fingerprints)
- Manhattan distinctive top-5: wallet, bag, clothing, electronics, insurance
  (lift around 4x baseline)
- Staten Island distinctive top-5: plowed, recy, ewaste, snow, route (lift up
  to 5.56x for "plowed" -- highest single-term lift in the corpus)
- Bronx, Brooklyn, Queens have less distinctive vocabularies overall

### Phase 10 (BERT MiniLM)
- Encoded 100K stratified subsample to 384-dim
- Best k matched to Phase 5: silhouette ~0.6881 at k=30
- Word2Vec at the same k: ~0.455
- BERT advantage: +0.233 absolute, +51% relative
- The "rat infestation in the building" probe should retrieve mostly
  Heat/Hot Water rows because the literal phrase "ENTIRE BUILDING" is the most
  frequent descriptor in the corpus. This is the documented honest finding -- do
  not be alarmed.

### Final
- Auto-commit cell prints something like `pushed 12 files to main, sha=<...>`.
- Check `https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp/commits/main`
  to confirm.
- Streamlit Cloud should pick up the push and rebuild within 5 minutes.

If all of the above lines up, the run is good.

---

## Quick reference: file roles

| File | Role |
|:-----|:-----|
| `notebooks/FINAL_PROJECT_311_NLP.ipynb` | The single end-to-end notebook |
| `src/spark_setup.py` | get_spark, reset_spark, parquet rebase mode handling |
| `src/ingest.py` | SODA paginated pull, .raw.parquet checkpoint, COLUMN_MAP normalization |
| `src/preprocess.py` | NLTK tokenize/stopword/lemmatize as a Spark Transformer |
| `src/classify.py` | TF-IDF + LR pipeline, portable .npz export, stratified split |
| `src/regress.py` | log1p target, v1 and v2 features, portable .npz export |
| `src/cluster.py` | Word2Vec + KMeans sweep + silhouette + top-terms per cluster |
| `src/geo.py` | Borough boundary load and lift computation |
| `src/colab_git.py` | commit_artifacts helper for auto-push |
| `dashboard/app.py` | Streamlit entry point |
| `dashboard/tabs/*.py` | One file per dashboard tab |
| `requirements.txt` | Dashboard runtime deps (Streamlit Cloud reads this) |
| `requirements-train.txt` | Training deps (Colab reads this; includes pyspark) |
| `runtime.txt` | Python version pin for Streamlit Cloud |
| `dashboard/assets/nyc_boroughs.geojson` | 5-polygon GeoJSON bundled in repo |

That's the whole project. If you got through all 11 phases and the dashboard is
live with the numbers above, you are done.
