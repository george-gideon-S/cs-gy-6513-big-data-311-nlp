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

This project applies distributed NLP at scale to the NYC 311 service request corpus (~43 million records since 2010). We classify free-text complaint descriptions into the right category, predict how long each will take to resolve, and surface latent groupings that cross the official taxonomy. The whole pipeline ends in a live Streamlit dashboard where you can paste a complaint and get back the triage result in under two seconds.

The work is built on Apache Spark (PySpark + MLlib) for batch training, Spark Structured Streaming for an incremental classification proof-of-concept, and Streamlit for the user-facing demo.

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
| `notebooks/` | Phase-by-phase Jupyter notebooks (00 env check through 10 BERT) |
| `src/` | Reusable Python modules imported by the notebooks |
| `dashboard/` | Streamlit app and per-tab modules |
| `models/portable/` | Spark-free model artifacts the deployed app loads |
| `docs/STEP_BY_STEP_GUIDE.md` | Micro-detailed runbook |
| `CLAUDE.md` | Project memory file (auto-loaded by Claude Code) |

## Datasets

- NYC 311 Service Requests (2020 to Present): https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9
- NYC 311 Service Requests (2010-2019): https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-2019/76ig-c548
- NYC Community Districts (GeoJSON): https://data.cityofnewyork.us/City-Government/Community-Districts/mzpm-a6vd
- U.S. Census ACS 5-Year Estimates: https://data.census.gov/

## Tech stack at a glance

| Layer | Tool |
|---|---|
| Compute | Google Colab Pro, PySpark in local mode |
| ML/NLP | Spark MLlib (TF-IDF, Word2Vec, LR, RF, K-Means, LDA), sentence-transformers |
| Storage | Parquet / Delta Lake on Drive |
| Streaming | Spark Structured Streaming (`Trigger.AvailableNow`) |
| Dashboard | Streamlit + Folium + Plotly |
| Hosting | Streamlit Community Cloud |

## License

MIT (or unspecified for now; will be added at submission time).
