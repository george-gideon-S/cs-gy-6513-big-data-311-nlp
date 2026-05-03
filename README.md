# The Language of Complaints
## NLP-Powered NYC 311 Service Request Triage and Resolution Time Prediction

**CS-GY 6513 Big Data, Final Project (Section D, Term 2)**
NYU Tandon School of Engineering, Prof. Amit Patel.

**Team:**
- George Gideon Sale (gs4602)
- Aayush Prranav Chandrashekar (ac11929)
- Shreeram Sankar (ss18731)

---

## What this project does

This project applies distributed NLP at scale to the NYC 311 service request corpus (~43 million records since 2010). We sample 10 million rows (5M from each of the city's two SODA endpoints, around 3 GB of raw CSV equivalent), classify free-text complaint descriptions into the right category, predict how long each will take to resolve, and surface latent groupings that cross the official taxonomy. The whole pipeline ends in a live Streamlit dashboard where you can paste a complaint and get back the triage result in under two seconds.

The work is built on Apache Spark (PySpark + MLlib) for batch training, Spark Structured Streaming for an incremental classification proof-of-concept, sentence-transformers MiniLM for a pretrained-embedding sidebar, and Streamlit for the user-facing demo.

## Live demo

The deployed dashboard lives at: *(to be added once deployed)*

If the public URL is down, the demo can also be brought up locally:

```bash
streamlit run dashboard/app.py
```

## Quick start (local development)

1. Clone the repo:
   ```bash
   git clone https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp.git
   cd cs-gy-6513-big-data-311-nlp
   ```
2. Create a Python 3.11 virtual environment and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Bring up the dashboard with the bundled portable models:
   ```bash
   streamlit run dashboard/app.py
   ```

For training from scratch on the full corpus, see [`docs/STEP_BY_STEP_GUIDE.md`](docs/STEP_BY_STEP_GUIDE.md). Heavy training is done on Google Colab Pro; the deployed app uses pre-exported portable artifacts so it can run on any Python host.

## Repository layout

| Path | What's there |
|---|---|
| `notebooks/FINAL_PROJECT_311_NLP.ipynb` | Single end-to-end notebook covering all phases |
| `src/` | Reusable Python modules imported by the notebook |
| `dashboard/` | Streamlit app and per-tab modules |
| `models/portable/` | Slim model artifacts the deployed dashboard loads (regenerated each notebook run) |
| `tools/build_final_notebook.py` | Build script that regenerates the unified notebook from a structured cell spec |
| `docs/STEP_BY_STEP_GUIDE.md` | Micro-detailed runbook for the unified notebook |
| `docs/REPORT.tex` | NeurIPS-format project report |

## Datasets

- NYC 311 Service Requests (2020 to Present): https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9 (about 20M rows)
- NYC 311 Service Requests (2010-2019 historical): https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/76ig-c548 (about 23M rows)
- NYC Borough Boundaries (GeoJSON): bundled in `dashboard/assets/nyc_boroughs.geojson` (5 polygons)

The combined 43M-row corpus is roughly 13 GB of raw CSV. We sample 10M rows (5M from each side) for a single end-to-end run that finishes in around 1 to 1.5 hours on Colab Pro with H100 or A100. The original community-districts dataset (`mzpm-a6vd`) was planned for finer-grained geographic aggregation but currently returns null geometries on the city's data portal, so we pivoted to the 5-borough level which is also more readable for a demo audience.

## Tech stack at a glance

| Layer | Tool |
|---|---|
| Compute | Google Colab Pro, H100 or A100 GPU, PySpark in local mode |
| Spark | PySpark 3.5 with Java 11 |
| ML/NLP | Spark MLlib (TF-IDF + Logistic Regression, Linear Regression, Word2Vec, K-Means), sentence-transformers MiniLM for the BERT comparison |
| Storage | Parquet on Drive, partitioned by year. Raw checkpoints written via pyarrow before any Spark conversion to make the pipeline crash-safe. |
| Streaming | On-demand SODA API pull in the dashboard's Live Pulse tab (Spark Structured Streaming PoC was scoped out of the deployed app) |
| Dashboard | Streamlit + Folium + matplotlib |
| Hosting | Streamlit Community Cloud, Python 3.11 |

## License

MIT (or unspecified for now; will be added at submission time).
