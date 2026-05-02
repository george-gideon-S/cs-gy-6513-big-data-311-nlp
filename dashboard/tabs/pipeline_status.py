"""
pipeline status - quick view of dataset stats, model versions, training metadata.
useful both for sanity-checking the deployment and for the live demo
"how do we know this is real" moment.
"""
from pathlib import Path
import streamlit as st


def _exists(rel: str) -> bool:
    return (Path(__file__).resolve().parents[2] / rel).exists()


def render() -> None:
    st.header("Pipeline Status")
    st.write(
        "Snapshot of where the project pipeline currently stands. Phases turn "
        "green as their artifacts land in the deployment."
    )

    rows = [
        ("Phase 0 - Environment", "—", True),
        ("Phase 1 - Ingest", "models/portable/", _exists("dashboard/assets/class_dist.png")),
        ("Phase 2 - Preprocess", "(in transformer)", _exists("dashboard/assets/class_dist.png")),
        ("Phase 3 - Classifier", "models/portable/classifier.npz", _exists("models/portable/classifier.npz")),
        ("Phase 4 - Regressor", "models/portable/regressor.npz", _exists("models/portable/regressor.npz")),
        ("Phase 5 - Word2Vec clusters", "models/portable/word2vec.kv", _exists("models/portable/word2vec.kv")),
        ("Phase 6 - Geo + Census join", "delta/district_fingerprints.parquet", _exists("delta/district_fingerprints.parquet")),
        ("Phase 8 - Streaming sink", "delta/streaming_out/", _exists("delta/streaming_out")),
        ("Phase 9 - LDA + trends", "delta/lda_topics.json", _exists("delta/lda_topics.json")),
        ("Phase 10 - BERT clusters", "delta/bert_clusters.parquet", _exists("delta/bert_clusters.parquet")),
    ]
    for name, artifact, ready in rows:
        icon = "✅" if ready else "⏳"
        st.markdown(f"{icon} **{name}** — `{artifact}`")
