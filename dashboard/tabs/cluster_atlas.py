"""
cluster atlas - word2vec vs bert side by side, plus per-district word clouds.
"""
import streamlit as st


def render() -> None:
    st.header("Cluster Atlas")
    st.write(
        "Latent groupings the model surfaces from the descriptions. Toggle "
        "between Word2Vec (Phase 5) and sentence-transformer BERT (Phase 10) "
        "embeddings to see how clusters change."
    )
    method = st.radio("Embedding method", ["Word2Vec", "BERT (MiniLM)"], horizontal=True)
    st.info(
        f"{method} clusters land in their respective phases. Once trained, "
        f"this tab will show the cluster list, top-10 terms per cluster, and "
        f"a per-district word cloud."
    )
