"""
spark structured streaming proof of concept.

we watch a folder for new csvs, classify each row using the trained pipeline
model, and write to a delta sink. the dashboards live pulse tab tails the
sink to show classifications arriving in near-real-time.

databricks free edition only supports Trigger.AvailableNow. we use that
trigger so the same code works in both environments.
"""
from __future__ import annotations
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType, DoubleType,
)


# input schema for csvs dropped into the watch folder. structured streaming
# requires an explicit schema for file sources.
WATCH_SCHEMA = StructType([
    StructField("unique_key", StringType(), True),
    StructField("created_date", TimestampType(), True),
    StructField("agency", StringType(), True),
    StructField("problem_detail", StringType(), True),
    StructField("borough", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
])


def start_classification_stream(
    spark: SparkSession,
    watch_folder: str | Path,
    sink_path: str | Path,
    checkpoint_path: str | Path,
    model: PipelineModel,
) -> StreamingQuery:
    """
    watch a folder for csvs, classify each row, append to delta sink.

    Args:
        spark: live spark session.
        watch_folder: directory we watch for new csvs.
        sink_path: where classified output lands (delta).
        checkpoint_path: where structured streaming stores its checkpoint
                         state. must be a fresh path on each pipeline reset.
        model: trained PipelineModel (output of phase 3 training).

    Returns:
        the StreamingQuery handle so callers can .stop() it.
    """
    from src.preprocess import TextPreprocessor

    watch_folder = str(watch_folder)
    sink_path = str(sink_path)
    checkpoint_path = str(checkpoint_path)

    Path(watch_folder).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # readStream from csv folder
    raw = (
        spark.readStream
        .schema(WATCH_SCHEMA)
        .option("header", "true")
        .option("maxFilesPerTrigger", 1)
        .csv(watch_folder)
    )

    # preprocess + classify - reusing the same transformer the batch model
    # was trained against
    pre = TextPreprocessor()
    tokenized = pre.transform(raw)
    classified = model.transform(tokenized)

    # keep only what the dashboard needs
    out = classified.select(
        "unique_key",
        "created_date",
        "agency",
        "borough",
        "latitude",
        "longitude",
        "problem_detail",
        "prediction",
        "probability",
    )

    # writeStream with Trigger.AvailableNow so this works on databricks free
    # edition AND on colab. it processes whatever files exist and stops.
    from pyspark.sql.streaming import Trigger
    query = (
        out.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
        .trigger(availableNow=True)
        .start(sink_path)
    )
    return query
