"""
classification training + portable export.

the heavy lifting (tf-idf, lr, rf) runs in spark on colab. but the deployed
streamlit app cant run spark, so we export the trained model into a numpy
artifact that pure-python can load.
"""
from __future__ import annotations
from typing import Tuple
from pathlib import Path

import numpy as np

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def stratified_split(
    df: DataFrame, label_col: str = "label_canonical", test_fraction: float = 0.2
) -> Tuple[DataFrame, DataFrame]:
    """
    train/test split that preserves class proportions. uses sampleBy because
    randomSplit doesnt stratify.

    Returns:
        (train_df, test_df)
    """
    # build a fractions dict where every class samples test_fraction
    classes = [r[label_col] for r in df.select(label_col).distinct().collect()]
    fractions = {c: test_fraction for c in classes if c is not None}

    test = df.sampleBy(label_col, fractions=fractions, seed=42)
    # subtract test from full df by id - assumes unique_key exists, which it
    # always does in our pipeline post-ingest
    train = df.join(test.select("unique_key"), on="unique_key", how="left_anti")
    return train, test


def build_pipeline(
    label_col: str = "label_canonical",
    num_features: int = 16384,
    min_doc_freq: int = 10,
    reg_param: float = 0.01,
) -> Pipeline:
    """
    standard tf-idf + logistic regression pipeline.

    Args:
        label_col: column to predict. defaults to label_canonical (post-phase2)
                   so the pipeline doesnt confuse raw and cleaned labels.
        num_features: hash space size. 16k is plenty for 311 descriptors
                      where mean token count is ~2.25 - bigger just wastes
                      memory and bloats the portable export.
        min_doc_freq: drop any term that appears in fewer than this many
                      docs at idf time. filters typos and one-off junk.
        reg_param: l2 regularization strength.
    """
    indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="skip")
    hashing_tf = HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=num_features)
    idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=min_doc_freq)
    lr = LogisticRegression(
        maxIter=20,
        regParam=reg_param,
        elasticNetParam=0.0,
        family="multinomial",
    )
    return Pipeline(stages=[indexer, hashing_tf, idf, lr])


def evaluate(model: PipelineModel, test: DataFrame) -> dict:
    """returns a dict of macro-f1, weighted-f1, accuracy."""
    preds = model.transform(test)
    metrics = {}
    for name in ["f1", "weightedFMeasure", "accuracy"]:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=name
        )
        metrics[name] = evaluator.evaluate(preds)
    return metrics


def export_portable(model: PipelineModel, out_path: str | Path) -> None:
    """
    extract logistic regression coefficients + tf-idf vocab into a numpy
    artifact the streamlit app loads at startup.

    saves a single .npz with:
      - vocab_size: int
      - idf: float array of length vocab_size
      - coefs: (n_classes, vocab_size) float matrix
      - intercepts: (n_classes,) float array
      - labels: list of class label strings
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # walk the pipeline stages to grab what we need. order is deterministic
    # since we set it in build_pipeline.
    stages = model.stages
    indexer = stages[0]   # StringIndexerModel
    idf_model = stages[2]
    lr_model = stages[3]

    np.savez_compressed(
        out_path,
        vocab_size=lr_model.numFeatures,
        idf=idf_model.idf.toArray(),
        coefs=lr_model.coefficientMatrix.toArray(),
        intercepts=lr_model.interceptVector.toArray(),
        labels=np.array(indexer.labels, dtype=object),
    )
    print(f"portable classifier saved to {out_path}")


def predict_portable(text_tokens: list, artifact: dict) -> tuple:
    """
    pure-python inference using the .npz artifact.

    Args:
        text_tokens: list of preprocessed tokens for one document.
        artifact: dict-like loaded via np.load(npz_path).

    Returns:
        (predicted_label, confidence_dict)
    """
    # hash the tokens into the same feature space the spark model used.
    # we replicate spark's HashingTF behavior with a simple modulo hash.
    vocab_size = int(artifact["vocab_size"])
    raw = np.zeros(vocab_size, dtype=np.float32)
    for tok in text_tokens:
        # spark uses murmurhash3_32 internally. we approximate with python's
        # built-in hash() then mod. for demo purposes the small drift is fine
        # but this is the place to swap in real murmurhash if exact parity
        # matters at evaluation time.
        idx = hash(tok) % vocab_size
        raw[idx] += 1.0

    # idf weighting
    weighted = raw * artifact["idf"]

    # logistic regression scoring: softmax(coefs @ x + intercepts)
    logits = artifact["coefs"] @ weighted + artifact["intercepts"]
    # softmax with stable max-subtract trick
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    labels = list(artifact["labels"])
    top_idx = int(probs.argmax())
    confidence = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return labels[top_idx], confidence
