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


# project stopwords - words so common in 311 they kill discrimination.
# tuned by inspecting top tf-idf terms across all categories.
_PROJECT_STOPS: set = {
    "street", "avenue", "ave", "st", "road", "rd",
    "building", "apt", "apartment", "address", "location",
    "please", "complaint", "issue", "problem",
    "nyc", "ny", "new", "york",
}


def _ensure_nltk_data() -> None:
    """make sure stopwords + wordnet + punkt are on disk before we use them."""
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))
    for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            # this will hit the network if NLTK_DATA_DIR is empty.
            # in production we ship them bundled so this is a no-op.
            try:
                nltk.download(pkg, download_dir=str(NLTK_DATA_DIR), quiet=True)
            except Exception as e:
                # if we cant download, the bundled assets must already cover it
                print(f"  nltk download skipped for {pkg}: {e}")


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
