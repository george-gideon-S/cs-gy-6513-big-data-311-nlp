"""
resolution time prediction.

target: hours from created_date to closed_date.
features: tf-idf (small vocab), agency one-hot, borough one-hot,
hour-of-day, day-of-week.

we report mae primarily because median resolution time has a long tail
and rmse over-penalizes the long-tail tickets we cant accurately predict
anyway.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    HashingTF, IDF, StringIndexer, OneHotEncoder, VectorAssembler,
)
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_resolution_label(df: DataFrame) -> DataFrame:
    """
    add `resolution_hours` column. drops rows that are null/negative which
    typically means the ticket is still open or has bad timestamp data.
    """
    return (
        df.withColumn(
            "resolution_hours",
            (F.unix_timestamp("closed_date") - F.unix_timestamp("created_date")) / 3600.0,
        )
        .filter(F.col("resolution_hours").isNotNull())
        .filter(F.col("resolution_hours") > 0)
        # cap absurd outliers - tickets open for >2 years are usually data bugs
        .filter(F.col("resolution_hours") < 24 * 365 * 2)
    )


def add_temporal_features(df: DataFrame) -> DataFrame:
    """add hour-of-day and day-of-week ints from created_date."""
    return (
        df.withColumn("hour_of_day", F.hour("created_date"))
        .withColumn("day_of_week", F.dayofweek("created_date"))
    )


def build_pipeline(num_features: int = 4096) -> Pipeline:
    """
    text features kept small here. resolution time is mostly determined by
    agency + borough + hour-of-day, with text giving marginal lift. we
    intentionally cap tf-idf small so the regression signal isnt drowned.
    """
    # indexers + onehot for categoricals
    agency_idx = StringIndexer(inputCol="agency", outputCol="agency_idx", handleInvalid="keep")
    borough_idx = StringIndexer(inputCol="borough", outputCol="borough_idx", handleInvalid="keep")
    agency_oh = OneHotEncoder(inputCol="agency_idx", outputCol="agency_vec")
    borough_oh = OneHotEncoder(inputCol="borough_idx", outputCol="borough_vec")

    # text features
    htf = HashingTF(inputCol="tokens", outputCol="raw_text", numFeatures=num_features)
    idf = IDF(inputCol="raw_text", outputCol="text_vec", minDocFreq=10)

    # combine everything
    assembler = VectorAssembler(
        inputCols=["text_vec", "agency_vec", "borough_vec", "hour_of_day", "day_of_week"],
        outputCol="features",
    )

    lr = LinearRegression(
        labelCol="resolution_hours",
        featuresCol="features",
        regParam=0.1,
        elasticNetParam=0.0,
        maxIter=50,
    )
    return Pipeline(
        stages=[agency_idx, borough_idx, agency_oh, borough_oh, htf, idf, assembler, lr]
    )


def evaluate(model: PipelineModel, test: DataFrame) -> dict:
    """returns mae, rmse, r2."""
    preds = model.transform(test)
    metrics = {}
    for name in ["mae", "rmse", "r2"]:
        evaluator = RegressionEvaluator(
            labelCol="resolution_hours", predictionCol="prediction", metricName=name
        )
        metrics[name] = evaluator.evaluate(preds)
    return metrics


def median_baseline_mae(train: DataFrame, test: DataFrame) -> float:
    """
    naive baseline: predict the median resolution hours per problem category.
    if our trained model cant beat this, the text features arent earning their keep.
    """
    medians = (
        train.groupBy("problem")
        .agg(F.expr("percentile_approx(resolution_hours, 0.5)").alias("median_hours"))
    )
    joined = test.join(medians, on="problem", how="left")
    # global median fallback for unseen categories
    global_median = train.approxQuantile("resolution_hours", [0.5], 0.01)[0]
    joined = joined.fillna({"median_hours": global_median})
    mae = joined.select(
        F.mean(F.abs(F.col("resolution_hours") - F.col("median_hours"))).alias("mae")
    ).collect()[0]["mae"]
    return float(mae)


def export_portable(model: PipelineModel, out_path: str | Path) -> None:
    """
    save just enough of the pipeline so the streamlit app can score in pure
    python. for the regressor we save the LR coefficients + the categorical
    indexers.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # find the LinearRegressionModel - its the last stage
    lr_model = model.stages[-1]
    np.savez_compressed(
        out_path,
        coefs=lr_model.coefficients.toArray(),
        intercept=float(lr_model.intercept),
        agency_labels=np.array(model.stages[0].labels, dtype=object),
        borough_labels=np.array(model.stages[1].labels, dtype=object),
    )
    print(f"portable regressor saved to {out_path}")
