"""
spark mllib lda topic modeling demo (phase 9).

trains lda over the preprocessed 311 corpus, prints top terms per topic,
saves a topic-term parquet for downstream notebooks, and emits a json
summary the dashboard can pick up.

usage:
    python tools/spark_lda_demo.py
    python tools/spark_lda_demo.py --k 20 --sample 100000
    python tools/spark_lda_demo.py --vocab-size 3000 --max-iter 30
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# project root on sys.path so `from src.* import ...` works
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _build_corpus(spark, parquet_path: str, sample_n: int):
    """load the preprocessed parquet, keep rows with >=2 tokens, sample."""
    from pyspark.sql import functions as F

    print(f"reading parquet at {parquet_path} ...")
    df = spark.read.parquet(parquet_path)
    total = df.count()
    print(f"  total rows: {total:,}")

    # rows must have a tokens array of length >= 2 to contribute to lda
    df = df.filter(F.col("tokens").isNotNull())
    df = df.filter(F.size(F.col("tokens")) >= 2)
    after_filter = df.count()
    print(f"  after >=2 tokens filter: {after_filter:,}")

    # sample down to the target count for tractability. fraction is calibrated
    # so we slightly oversample then limit, which is closer to the requested N
    # than fraction alone.
    if after_filter > sample_n:
        frac = min(1.0, (sample_n * 1.5) / after_filter)
        df = df.sample(False, frac, seed=42).limit(sample_n)

    df = df.select("tokens").cache()
    n = df.count()
    print(f"  sampled corpus size: {n:,}")
    return df, n


def _format_topics_table(topics):
    """turn list[{id, top_terms, share}] into an ascii table for stdout."""
    header = f"{'id':>3} | {'share':>7} | top terms"
    sep = "-" * 80
    out = [header, sep]
    for t in topics:
        terms = ", ".join(t["top_terms"][:10])
        out.append(f"{t['id']:>3} | {t['share']:>7.3f} | {terms}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="spark mllib lda topic modeling demo")
    ap.add_argument("--k", type=int, default=15, help="number of topics (default 15)")
    ap.add_argument("--sample", type=int, default=200_000,
                    help="rows to sample for tractable training (default 200k)")
    ap.add_argument("--vocab-size", type=int, default=2000,
                    help="CountVectorizer vocab size (default 2000)")
    ap.add_argument("--min-df", type=int, default=10,
                    help="CountVectorizer minDF (default 10)")
    ap.add_argument("--max-iter", type=int, default=20,
                    help="lda max iterations (default 20)")
    ap.add_argument("--parquet", type=str, default=None,
                    help="override preprocessed parquet path")
    ap.add_argument("--top-terms", type=int, default=10,
                    help="how many top terms to surface per topic (default 10)")
    args = ap.parse_args()

    from src.spark_setup import get_spark
    from src.config import ROOT, ASSETS_DIR
    from pyspark.ml.clustering import LDA
    from pyspark.ml.feature import CountVectorizer
    from pyspark.sql import functions as F

    parquet_path = args.parquet or str(ROOT / "sample_2m_preprocessed.parquet")
    if not Path(parquet_path).exists():
        raise FileNotFoundError(
            f"preprocessed parquet not found at {parquet_path}. "
            f"run the main notebook first or pass --parquet."
        )

    t0 = time.time()
    spark = get_spark(app_name="spark-lda-demo")
    print(f"spark up in {time.time() - t0:.1f}s")

    df, n_docs = _build_corpus(spark, parquet_path, args.sample)

    # CountVectorizer fits the vocab. minDF=10 drops typos / one-off tokens.
    print(f"\nfitting CountVectorizer (vocabSize={args.vocab_size}, minDF={args.min_df}) ...")
    t = time.time()
    cv = CountVectorizer(
        inputCol="tokens",
        outputCol="bow",
        vocabSize=args.vocab_size,
        minDF=float(args.min_df),
    )
    cv_model = cv.fit(df)
    bow = cv_model.transform(df).cache()
    vocab = list(cv_model.vocabulary)
    print(f"  vocab learned: {len(vocab):,} terms in {time.time() - t:.1f}s")

    # LDA. seed makes the run reproducible.
    print(f"\nfitting LDA (k={args.k}, maxIter={args.max_iter}) ...")
    t = time.time()
    lda = LDA(
        featuresCol="bow",
        k=args.k,
        maxIter=args.max_iter,
        seed=42,
    )
    lda_model = lda.fit(bow)
    print(f"  lda fit in {time.time() - t:.1f}s")

    # describeTopics gives us (termIndices, termWeights) per topic.
    desc = lda_model.describeTopics(maxTermsPerTopic=args.top_terms).collect()

    # to compute topic share we average the per-doc topic distribution. transform
    # adds a `topicDistribution` column - a dense vector of length k. we sum and
    # normalize.
    print("\ncomputing dominant-topic share ...")
    transformed = lda_model.transform(bow)
    # use a small udf to expose the vector entries as an array of doubles so
    # we can aggregate them column-wise via posexplode.
    from pyspark.sql.types import ArrayType, DoubleType

    @F.udf(returnType=ArrayType(DoubleType()))
    def _vec_to_arr(v):
        try:
            return [float(x) for x in v.toArray()]
        except Exception:
            return []

    arr_df = transformed.select(_vec_to_arr(F.col("topicDistribution")).alias("td"))
    agg = (
        arr_df.select(F.posexplode("td").alias("topic_id", "weight"))
        .groupBy("topic_id")
        .agg(F.sum("weight").alias("mass"))
        .orderBy("topic_id")
        .collect()
    )
    total_mass = float(sum(r["mass"] for r in agg)) or 1.0
    share_by_topic = {int(r["topic_id"]): float(r["mass"]) / total_mass for r in agg}

    # build the json-friendly topics list
    topics = []
    for row in desc:
        tid = int(row["topic"])
        idxs = list(row["termIndices"])
        terms = [vocab[i] for i in idxs if 0 <= i < len(vocab)]
        topics.append({
            "id": tid,
            "top_terms": terms,
            "share": round(share_by_topic.get(tid, 0.0), 4),
        })
    # sort the report by descending share so the dominant topics surface first
    topics_sorted = sorted(topics, key=lambda t: -t["share"])

    print("\n" + _format_topics_table(topics_sorted))

    # save the full topic-term matrix as parquet. shape is (k * top_terms)
    # rows so other notebooks can join or rank.
    rows = []
    for row in desc:
        tid = int(row["topic"])
        idxs = list(row["termIndices"])
        weights = list(row["termWeights"])
        for rank, (i, w) in enumerate(zip(idxs, weights)):
            term = vocab[i] if 0 <= i < len(vocab) else None
            rows.append((tid, rank, term, float(w)))
    topic_term_df = spark.createDataFrame(rows, ["topic_id", "rank", "term", "weight"])
    topic_parquet = ROOT / "lda_topics.parquet"
    print(f"\nwriting topic-term parquet to {topic_parquet} ...")
    topic_term_df.coalesce(1).write.mode("overwrite").parquet(str(topic_parquet))

    # save the json summary the dashboard can read.
    summary = {
        "phase": 9,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_docs": int(n_docs),
        "k_topics": int(args.k),
        "vocab_size": int(len(vocab)),
        "min_df": int(args.min_df),
        "max_iter": int(args.max_iter),
        "topics": topics_sorted,
    }
    assets_dir = Path(ASSETS_DIR)
    assets_dir.mkdir(parents=True, exist_ok=True)
    json_path = assets_dir / "lda_topics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote dashboard summary to {json_path}")

    print("\n=== SPARK LDA COMPLETE ===")


if __name__ == "__main__":
    main()
