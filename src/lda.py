"""
lda topic modeling + sliding-window trend detection.

lda surfaces themes that cross the official problem taxonomy. trend
detection flags terms whose recent (last-30-day) frequency is unusually
high vs the historical average - useful for spotting emerging issues
before they make it into the category counts.
"""
from __future__ import annotations
from typing import List

import numpy as np

from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def fit_lda(df: DataFrame, k: int = 25, max_iter: int = 20) -> tuple:
    """
    fit lda. returns (cv_model, lda_model). the count vectorizer model is
    needed for translating topic indices back to terms.

    Args:
        df: must have `tokens` column.
        k: number of topics.
        max_iter: lda iterations. 20 usually plenty for our corpus.
    """
    cv = CountVectorizer(inputCol="tokens", outputCol="bow", vocabSize=10_000, minDF=10.0)
    cv_model = cv.fit(df)
    bow = cv_model.transform(df)

    lda = LDA(featuresCol="bow", k=k, maxIter=max_iter, seed=42)
    lda_model = lda.fit(bow)
    return cv_model, lda_model


def topic_terms(
    cv_model: CountVectorizerModel, lda_model: LDAModel, n_terms: int = 10
) -> List[List[str]]:
    """
    returns a list of length k where each item is the top n_terms terms
    for that topic.
    """
    vocab = cv_model.vocabulary
    topics = lda_model.describeTopics(maxTermsPerTopic=n_terms).collect()
    out = []
    for row in topics:
        out.append([vocab[i] for i in row["termIndices"]])
    return out


def detect_term_spikes(
    df: DataFrame,
    window_days: int = 30,
    z_threshold: float = 3.0,
    min_recent_count: int = 50,
) -> List[dict]:
    """
    find terms whose recent (last `window_days`) frequency is `z_threshold`
    standard deviations above their long-term average.

    Args:
        df: must have `tokens` and `created_date`.
        window_days: how recent counts as "recent".
        z_threshold: how many sd above mean to flag.
        min_recent_count: drop terms with too-few recent occurrences to
                          avoid flagging noise.

    Returns:
        list of dicts with term, recent_count, historical_mean, z_score.
    """
    # latest date in corpus
    max_date = df.select(F.max("created_date")).collect()[0][0]
    recent_cutoff = F.lit(max_date) - F.expr(f"INTERVAL {window_days} DAYS")

    exploded = df.select("created_date", F.explode("tokens").alias("term"))

    # recent counts per term
    recent = (
        exploded.filter(F.col("created_date") >= recent_cutoff)
        .groupBy("term").count()
        .withColumnRenamed("count", "recent_count")
        .filter(F.col("recent_count") >= min_recent_count)
    )

    # historical per-window counts. we approximate the historical distribution
    # by counting per 30-day window and computing mean+sd.
    bucketed = exploded.withColumn(
        "window_id", (F.unix_timestamp("created_date") / (60 * 60 * 24 * window_days)).cast("int")
    )
    per_window = bucketed.groupBy("term", "window_id").count()
    stats = per_window.groupBy("term").agg(
        F.mean("count").alias("hist_mean"),
        F.stddev("count").alias("hist_sd"),
    )

    joined = recent.join(stats, on="term", how="left").fillna({"hist_sd": 1.0})
    joined = joined.withColumn(
        "z_score",
        (F.col("recent_count") - F.col("hist_mean")) / F.col("hist_sd"),
    ).filter(F.col("z_score") > z_threshold)

    return [r.asDict() for r in joined.orderBy(F.desc("z_score")).limit(50).collect()]
