"""
word2vec embeddings + kmeans clustering on 311 descriptions.

we train w2v over the tokenized corpus, get a doc embedding by averaging
word vectors per row (spark mllib's word2vec model.transform does this for
us), then sweep kmeans k for silhouette and pick the best.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def train_word2vec(
    df: DataFrame, vector_size: int = 100, window: int = 5, min_count: int = 5
) -> Word2VecModel:
    """
    train word2vec on the `tokens` column. vector_size 100 is a sweet spot
    for ~2m short documents - bigger doesnt help on this corpus and bloats
    the model file.
    """
    w2v = Word2Vec(
        vectorSize=vector_size,
        windowSize=window,
        minCount=min_count,
        inputCol="tokens",
        outputCol="doc_vec",
        seed=42,
    )
    return w2v.fit(df)


def sweep_kmeans(
    df_with_vecs: DataFrame,
    k_min: int = 5,
    k_max: int = 30,
    step: int = 5,
) -> List[Tuple[int, float, KMeansModel]]:
    """
    fit kmeans for k in [k_min, k_max] stepping by `step`, return
    list of (k, silhouette, model) sorted by k.
    """
    evaluator = ClusteringEvaluator(featuresCol="doc_vec", metricName="silhouette")
    out = []
    for k in range(k_min, k_max + 1, step):
        km = KMeans(featuresCol="doc_vec", predictionCol="cluster", k=k, seed=42)
        model = km.fit(df_with_vecs)
        preds = model.transform(df_with_vecs)
        score = evaluator.evaluate(preds)
        out.append((k, score, model))
        print(f"  k={k:>3}  silhouette={score:.4f}")
    return out


def top_terms_per_cluster(
    df: DataFrame, n_terms: int = 10
) -> dict:
    """
    surface the most-frequent tokens per cluster. cheap proxy for "what
    is this cluster about" - good enough for the demo.

    Args:
        df: must have `cluster` and `tokens` columns.
        n_terms: how many top terms per cluster.
    """
    # explode tokens, count per (cluster, term), keep top n per cluster
    exploded = df.select("cluster", F.explode("tokens").alias("term"))
    counts = exploded.groupBy("cluster", "term").count()

    # window function-free approach - just pull the data to pandas since
    # were dealing with at most ~30 clusters * a few thousand unique terms
    pdf = counts.toPandas()
    out = {}
    for cluster_id, sub in pdf.groupby("cluster"):
        top = sub.nlargest(n_terms, "count")
        out[int(cluster_id)] = list(zip(top["term"], top["count"]))
    return out


def export_word2vec(w2v_model: Word2VecModel, out_path: str | Path) -> None:
    """
    save word2vec as gensim KeyedVectors so the streamlit app can use it
    without spark. spark's word2vec gives us a `getVectors` dataframe we
    convert to a numpy table.
    """
    import numpy as np
    import gensim

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # spark gives us a df with columns word, vector (DenseVector)
    vec_df = w2v_model.getVectors().toPandas()
    words = vec_df["word"].tolist()
    vectors = np.stack([v.toArray() for v in vec_df["vector"]])

    # build a gensim KeyedVectors object and save
    kv = gensim.models.KeyedVectors(vector_size=vectors.shape[1])
    kv.add_vectors(words, vectors)
    kv.save(str(out_path))
    print(f"word2vec saved as gensim KeyedVectors to {out_path}")
