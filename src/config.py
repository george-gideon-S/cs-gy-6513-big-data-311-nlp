"""
project-wide constants. paths default to colab + drive layout but
fall back to a local working dir if drive isnt mounted.
"""
from pathlib import Path
import os

# random seed used everywhere for reproducibility
SEED = 42

# how many rows we sample for the dev corpus.
# professor wants >=1m. on colab with H100/A100 we run the FULL corpus
# (~43M rows). set SAMPLE_SIZE to None to skip the stratified-sample step
# entirely and use whatever the ingest produced.
SAMPLE_SIZE = None  # full corpus mode; was 2_000_000 during dev

# paths
# on colab we mount drive at /content/drive. when we dev locally on the
# windows box we just write into ./data and ./delta.
_DRIVE_ROOT = Path("/content/drive/MyDrive/cs6513")
_LOCAL_ROOT = Path(__file__).resolve().parents[1]

def _root() -> Path:
    """returns drive root if were on colab with drive mounted, else local."""
    if _DRIVE_ROOT.exists():
        return _DRIVE_ROOT
    return _LOCAL_ROOT

ROOT = _root()
DATA_DIR = ROOT / "data"
DELTA_DIR = ROOT / "delta"
MODELS_DIR = ROOT / "models"
PORTABLE_DIR = MODELS_DIR / "portable"
ASSETS_DIR = _LOCAL_ROOT / "dashboard" / "assets"

# datasets - soda api endpoints
SODA_2020_PLUS = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
SODA_2010_2019 = "https://data.cityofnewyork.us/resource/76ig-c548.csv"
SODA_DISTRICTS = "https://data.cityofnewyork.us/resource/jp9i-3b7y.geojson"

# top-k complaint categories we model. anything outside this falls into "Other"
# during classification training so we dont waste capacity on long-tail classes
# that have <1k records each.
TOP_K_CATEGORIES = 20

# nltk data dir (we bundle these to avoid runtime downloads on locked networks)
NLTK_DATA_DIR = ASSETS_DIR / "nltk_data"

# soda app token - read from env var if set, else None (anonymous calls work
# but throttle hard above ~1k requests/hour)
SODA_APP_TOKEN = os.environ.get("SODA_APP_TOKEN")
