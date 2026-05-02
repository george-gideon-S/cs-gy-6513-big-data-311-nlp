"""
triage bot - the centerpiece tab.

paste a complaint -> classify + predict resolution + nearest cluster +
nearest neighbors + map pin. runs the portable classifier and regressor
loaded from models/portable. spark not required at serve time.
"""
from pathlib import Path
import streamlit as st


def _load_portable_classifier():
    """try to load the portable classifier .npz. returns None if not yet trained."""
    import numpy as np
    p = Path(__file__).resolve().parents[2] / "models" / "portable" / "classifier.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def render() -> None:
    st.header("Triage Bot")
    st.write(
        "Paste a real or imagined NYC 311 complaint description below. "
        "The system will classify it, predict how long the city will take to "
        "resolve it, place it in a complaint cluster, and surface similar "
        "historical complaints from a 2 million row corpus."
    )

    sample = (
        "There's a really bad smell coming from the trash piles on 3rd Ave, "
        "been like this for days, rats everywhere now."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area(
            "Complaint description",
            value=st.session_state.get("triage_text", sample),
            height=140,
            key="triage_text",
        )
    with col2:
        st.selectbox(
            "Borough",
            ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"],
            key="triage_borough",
        )
        st.text_input("Zip code (optional)", key="triage_zip")
        run = st.button("Triage this complaint", type="primary", use_container_width=True)

    if not run:
        st.info("Click 'Triage this complaint' to see results.")
        return

    # try to load model. if missing, show a friendly placeholder.
    artifact = _load_portable_classifier()
    if artifact is None:
        st.warning(
            "The classifier hasn't been trained yet. Run "
            "`notebooks/03_classify.ipynb` and export the portable artifact "
            "to `models/portable/classifier.npz` to enable live predictions."
        )
        return

    # do the inference
    from src.preprocess import preprocess_text, _english_stops
    from src.classify import predict_portable

    tokens = preprocess_text(text, _english_stops())
    label, conf = predict_portable(tokens, artifact)

    st.subheader(f"Predicted category: {label}")
    # show top 5 confidences as a bar chart
    top5 = sorted(conf.items(), key=lambda kv: kv[1], reverse=True)[:5]
    chart_data = {"category": [k for k, _ in top5], "confidence": [v for _, v in top5]}
    st.bar_chart(chart_data, x="category", y="confidence", height=240)

    # placeholders for the other panels until phases 4-6 land
    cols = st.columns(3)
    with cols[0]:
        st.metric("Predicted resolution time", "(coming - phase 4)")
    with cols[1]:
        st.metric("Closest cluster", "(coming - phase 5)")
    with cols[2]:
        st.metric("Similar past complaints", "(coming - phase 6)")
