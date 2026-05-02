"""
text preprocessing for 311 complaint descriptions.

we lowercase, tokenize on non-alpha boundaries, drop nltk english stopwords
plus a small project-specific stoplist (think 'street', 'building' which
appear in nearly every complaint and dont help discriminate categories),
and lemmatize with wordnet so plurals/tenses collapse.

the spark transformer wrapper makes this slot into a pipeline so we can
serialize preprocessing alongside the trained classifier.
"""
from __future__ import annotations
from typing import List, Iterable
import re

import nltk
from nltk.stem import WordNetLemmatizer

from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from src.config import NLTK_DATA_DIR


# label canonicalization map. the historical 2010-19 dataset uses ALL-CAPS
# category names while 2020+ uses Title Case. some categories were also
# renamed in the 2018-19 taxonomy refresh.
#
# left side = name as it appears in raw data (lowercased for case-insensitive
# matching). right side = canonical name we use for training.
#
# tuned from inspecting our top-20 distribution. extend as new mismatches
# are discovered.
LABEL_CANONICAL_MAP: dict = {
    # heat related
    "heating": "Heat/Hot Water",
    "heat/hot water": "Heat/Hot Water",
    "hot water": "Heat/Hot Water",
    # plumbing
    "plumbing": "Plumbing",
    # painting
    "paint - plaster": "Paint/Plaster",
    "paint/plaster": "Paint/Plaster",
    # construction
    "general construction": "General Construction",
    "general construction/plumbing": "General Construction",
    # noise variants - the 2020+ data already splits these well, just unify casing
    "noise - residential": "Noise - Residential",
    "noise - street/sidewalk": "Noise - Street/Sidewalk",
    "noise - commercial": "Noise - Commercial",
    "noise - vehicle": "Noise - Vehicle",
    "noise - park": "Noise - Park",
    # street stuff
    "street condition": "Street Condition",
    "street light condition": "Street Light Condition",
    "street sign - missing": "Street Sign - Missing",
    "street sign - damaged": "Street Sign - Damaged",
    # parking
    "illegal parking": "Illegal Parking",
    "blocked driveway": "Blocked Driveway",
    # sanitation
    "request large bulky item collection": "Bulky Item Collection",
    "bulky item collection": "Bulky Item Collection",
    "dirty conditions": "Dirty Conditions",
    "missed collection (all materials)": "Missed Collection",
    "missed collection": "Missed Collection",
    # housing
    "unsanitary condition": "Unsanitary Condition",
    "rodent": "Rodent",
    # police
    "non-emergency police matter": "Non-Emergency Police Matter",
    # water
    "water system": "Water System",
}


def canonicalize_label(raw: str) -> str:
    """
    map a raw 311 problem string to its canonical form.
    falls through to the original string if no mapping exists,
    so unknown categories arent silently dropped.
    """
    if raw is None:
        return None
    key = raw.strip().lower()
    return LABEL_CANONICAL_MAP.get(key, raw.strip())


def add_canonical_label(df: DataFrame, in_col: str = "problem", out_col: str = "label_canonical") -> DataFrame:
    """spark wrapper around canonicalize_label, adds an extra column."""

    @F.udf(returnType=StringType())
    def _udf(s):
        return canonicalize_label(s)

    return df.withColumn(out_col, _udf(F.col(in_col)))


# project stopwords - words so common in 311 they kill discrimination.
# tuned by inspecting top tf-idf terms across all categories.
_PROJECT_STOPS: set = {
    "street", "avenue", "ave", "st", "road", "rd",
    "building", "apt", "apartment", "address", "location",
    "please", "complaint", "issue", "problem",
    "nyc", "ny", "new", "york",
}


# where to look for nltk data. order matters: project-bundled first, then
# system path that spark workers find via their default search.
#
# the second path is the important one for spark - workers receive src/ via
# addPyFile but cant see dashboard/assets/, so they need a system-wide
# nltk path they can fall back on. /root/nltk_data is one of nltks default
# search paths so this works without per-worker config.
_NLTK_PATHS = [str(NLTK_DATA_DIR), "/root/nltk_data", "/usr/share/nltk_data"]


def _ensure_nltk_data() -> None:
    """make sure stopwords + wordnet + punkt are on disk before we use them."""
    for p in _NLTK_PATHS:
        if p not in nltk.data.path:
            nltk.data.path.insert(0, p)

    for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
        try:
            nltk.data.find(pkg if "/" in pkg else f"corpora/{pkg}")
        except LookupError:
            # download to /root/nltk_data which workers will find too.
            # falling back to the project assets dir if /root isnt writable.
            for target in ["/root/nltk_data", str(NLTK_DATA_DIR)]:
                try:
                    nltk.download(pkg, download_dir=target, quiet=True)
                    break
                except Exception:
                    continue


def _english_stops() -> set:
    """returns nltk's english stops merged with our project stops."""
    _ensure_nltk_data()
    from nltk.corpus import stopwords
    return set(stopwords.words("english")) | _PROJECT_STOPS


# regex compiled once at module load. matches anything that isnt an
# english letter so contractions like "don't" become "don" and "t".
# this is fine for our purposes since both halves get stopworded out anyway.
_TOKEN_RE = re.compile(r"[^a-zA-Z]+")


# we instantiate the lemmatizer once. wordnet's first call triggers a load
# so the first row is slow but every subsequent row is cheap.
_LEMMATIZER = None


def _lemmatizer() -> WordNetLemmatizer:
    global _LEMMATIZER
    if _LEMMATIZER is None:
        _ensure_nltk_data()
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def preprocess_text(text: str, stops: set) -> List[str]:
    """
    pure-python preprocessor used inside the spark udf and also at inference
    time in the streamlit app. keeps train/test paths identical.
    """
    if text is None:
        return []
    lower = text.lower()
    tokens = [t for t in _TOKEN_RE.split(lower) if t]
    lem = _lemmatizer()
    out = []
    for t in tokens:
        if len(t) < 3:
            # one and two letter tokens are usually noise
            continue
        if t in stops:
            continue
        out.append(lem.lemmatize(t))
    return out


class TextPreprocessor(Transformer):
    """
    spark transformer that adds a `tokens` column. unfit-able since theres
    no learned state.

    Args:
        input_col: source column with raw description text.
        output_col: where the token list goes (default "tokens").
    """

    def __init__(self, input_col: str = "problem_detail", output_col: str = "tokens"):
        super().__init__()
        self.input_col = input_col
        self.output_col = output_col

    def _transform(self, df: DataFrame) -> DataFrame:
        # we close over the stopword set so it travels to executors as part
        # of the udf closure. its small (~200 strings).
        stops = _english_stops()

        @F.udf(returnType=ArrayType(StringType()))
        def _tok(text):
            return preprocess_text(text, stops)

        return df.withColumn(self.output_col, _tok(F.col(self.input_col)))
