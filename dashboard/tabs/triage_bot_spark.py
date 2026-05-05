"""
triage bot - spark edition.

alternate streamlit tab that performs end-to-end inference using apache
spark mllib (HashingTF + IDF + LogisticRegression) instead of the portable
numpy fast path used by the deployed Triage Bot tab.

when this is useful:
  - running locally: `streamlit run dashboard/app.py` on a box with java +
    pyspark installed. you get authentic spark mllib at serve time so the
    hash buckets are exact-match with training (spark MurmurHash3, seed=42)
    and the confidence numbers carry zero approximation overhead.

when this gracefully degrades:
  - streamlit cloud and any other no-java deploy. pyspark is intentionally
    omitted from requirements.txt to keep the slim deploy small. in that
    environment this tab renders an explanatory banner and the existing
    Triage Bot tab handles inference via numpy.

note: this tab is not registered in dashboard/app.py. it is an additive
component the user can wire in later.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st


# project root for imports / model paths. dashboard/tabs/triage_bot_spark.py
# is two levels deep, so parents[2] is the repo root.
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _try_import_pyspark():
    """attempt the pyspark import. return (ok, error_message)."""
    try:
        import pyspark  # noqa: F401
        return True, None
    except Exception as exc:
        return False, str(exc)


@st.cache_resource
def _spark_session():
    """one spark session per streamlit process. cached so reruns reuse it."""
    from src.spark_setup import get_spark
    return get_spark(app_name="dashboard-triage-bot-spark")


@st.cache_resource
def _classifier_model(model_dir: str):
    """load the saved PipelineModel once, share across reruns."""
    # ensure the spark session exists before we try to load the model
    _ = _spark_session()
    from pyspark.ml import PipelineModel
    return PipelineModel.load(model_dir)


def _resolve_classifier_labels(pipeline_model):
    """find the StringIndexerModel inside the pipeline and return its labels."""
    for stage in pipeline_model.stages:
        if hasattr(stage, "labels") and isinstance(stage.labels, list):
            return list(stage.labels)
    return []


def _format_pipeline_stages(pipeline_model) -> str:
    parts = []
    for stage in pipeline_model.stages:
        cls = type(stage).__name__
        if cls.endswith("Model"):
            cls = cls[: -len("Model")]
        parts.append(cls)
    return " -> ".join(parts)


def _resolve_model_dir() -> Path:
    """canonical classifier_lr path. uses src.config.MODELS_DIR like
    everything else in the project."""
    from src.config import MODELS_DIR
    return Path(MODELS_DIR) / "classifier_lr"


def _classify_with_spark(text: str, borough: str, classifier_model):
    """run a single-row spark dataframe through the classifier and return the
    ranked (label, prob) list plus the predicted top-1 string."""
    spark = _spark_session()
    from pyspark.sql.types import StructType, StructField, StringType
    from src.preprocess import add_canonical_label, TextPreprocessor

    schema = StructType([
        StructField("unique_key", StringType(), True),
        StructField("problem", StringType(), True),
        StructField("problem_detail", StringType(), True),
        StructField("agency", StringType(), True),
        StructField("borough", StringType(), True),
    ])
    rows = [("ui-0", "Heat/Hot Water", text, "HPD", borough.upper())]
    df = spark.createDataFrame(rows, schema)
    df = add_canonical_label(df, in_col="problem", out_col="label_canonical")
    df = TextPreprocessor(input_col="problem_detail", output_col="tokens").transform(df)

    scored = classifier_model.transform(df)
    row = scored.select("probability").collect()[0]
    probs = list(row["probability"].toArray())

    labels = _resolve_classifier_labels(classifier_model)
    pairs = list(zip(labels, [float(p) for p in probs]))
    pairs.sort(key=lambda kv: -kv[1])
    return pairs


def _render_unavailable(error_message: str | None):
    """no-pyspark path: show a clean explanatory banner and stop."""
    st.header("Triage Bot (Spark)")
    st.markdown(
        '<div style="margin-top: 8px; padding: 16px 20px; '
        'background: #FAFAFC; border-left: 3px solid #9B6FC2; '
        'border-radius: 4px; font-size: 14px; color: #4A4A4A; line-height: 1.55;">'
        "<b style=\"color: #57068C;\">Spark inference requires a local install with Java.</b> "
        "this tab uses apache spark mllib end-to-end and needs a jvm at "
        "serve time. streamlit community cloud does not ship java, so "
        "pyspark is intentionally omitted from <code>requirements.txt</code>. "
        "for the no-java fast path see the standard <b>Triage Bot</b> tab, "
        "which uses numpy with the same MurmurHash3 buckets the trained spark "
        "model learned (via <code>mmh3</code>, seed=42)."
        "</div>",
        unsafe_allow_html=True,
    )
    if error_message:
        with st.expander("import error detail"):
            st.code(error_message)
    st.caption(
        "to run this tab locally: install pyspark + java 11, then "
        "`streamlit run dashboard/app.py` from the project root."
    )


def render() -> None:
    ok, err = _try_import_pyspark()
    if not ok:
        _render_unavailable(err)
        return

    st.header("Triage Bot (Spark)")
    st.markdown(
        '<p style="font-size: 15px; color: #4A4A4A; line-height: 1.55; '
        'max-width: 800px;">'
        "this tab classifies an NYC 311 complaint using apache spark mllib "
        "end-to-end. hash buckets are exact-match with training "
        "(spark MurmurHash3, seed=42). the standard Triage Bot tab uses "
        "numpy with mmh3 for sub-second response on streamlit cloud's "
        "no-java environment."
        "</p>",
        unsafe_allow_html=True,
    )

    model_dir = _resolve_model_dir()
    if not model_dir.exists():
        st.warning(
            f"classifier_lr PipelineModel not found at `{model_dir}`. "
            f"train it via the main notebook first."
        )
        return

    sample = (
        "There's a really bad smell coming from the trash piles on 3rd Ave, "
        "been like this for days, rats everywhere now."
    )

    col_input, col_meta = st.columns([3, 1])
    with col_input:
        text = st.text_area(
            "Complaint description",
            value=st.session_state.get("triage_spark_text", sample),
            height=140,
            key="triage_spark_text",
            label_visibility="visible",
        )
    with col_meta:
        borough = st.selectbox(
            "Borough",
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
            index=0,
            key="triage_spark_borough",
        )
        st.write("")
        run = st.button(
            "Triage with Spark", type="primary", use_container_width=True,
            key="triage_spark_run",
        )

    if not run:
        st.markdown(
            '<div style="margin-top: 18px; padding: 14px 18px; '
            'background: #F5F2F8; border-left: 3px solid #9B6FC2; '
            'border-radius: 4px; font-size: 14px; color: #4A4A4A;">'
            "press <b style=\"color: #57068C;\">Triage with Spark</b> to "
            "send the text through the live spark mllib pipeline."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if not text or not text.strip():
        st.warning("please paste a complaint description first.")
        return

    # warm spark + load model on first click. both are @st.cache_resource so
    # subsequent clicks reuse the warmed state.
    with st.spinner("warming spark + loading PipelineModel ..."):
        try:
            classifier_model = _classifier_model(str(model_dir))
        except Exception as exc:
            st.error(f"failed to load classifier_lr: {exc}")
            return

    with st.spinner("scoring with spark mllib ..."):
        try:
            ranked = _classify_with_spark(text, borough, classifier_model)
        except Exception as exc:
            st.error(f"spark inference failed: {exc}")
            return

    if not ranked:
        st.warning("no probability vector returned. try a longer description.")
        return

    label, prob = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else (None, 0.0)

    st.markdown(
        f'<div style="margin-top: 8px; padding: 18px 22px; '
        f'background: linear-gradient(135deg, #57068C 0%, #3A0560 100%); '
        f'border-radius: 8px; color: white;">'
        f'<div style="font-size: 12px; font-weight: 600; '
        f'text-transform: uppercase; letter-spacing: 0.08em; '
        f'opacity: 0.85;">Predicted category (Spark)</div>'
        f'<div style="font-size: 30px; font-weight: 700; '
        f'margin-top: 4px;">{label}</div>'
        f'<div style="font-size: 14px; opacity: 0.85; margin-top: 6px;">'
        f"Confidence {prob * 100:.1f}% &nbsp;&middot;&nbsp; "
        f"Runner-up: {runner_up[0]} ({runner_up[1] * 100:.1f}%)"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Top 5 confidences (Spark)")
    top5 = ranked[:5]
    try:
        from dashboard.charts import confidence_bars
        fig = confidence_bars(
            [k for k, _ in top5],
            [v for _, v in top5],
            height=240,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # fallback if charts module unavailable - keep this self-contained
        for lab, p in top5:
            st.write(f"{lab}: {p * 100:.1f}%")

    # spark engine info footer - the whole point of this tab is to be honest
    # about what is running underneath.
    spark = _spark_session()
    sc = spark.sparkContext
    st.markdown("### Spark engine info")
    st.markdown(
        f'<div style="padding: 14px 18px; background: #FAFAFC; '
        f'border-left: 3px solid #57068C; border-radius: 4px; '
        f'font-size: 13px; color: #4A4A4A; line-height: 1.7; '
        f'font-family: Menlo, monospace;">'
        f"spark version       : {spark.version}<br>"
        f"master              : {sc.master}<br>"
        f"default parallelism : {sc.defaultParallelism}<br>"
        f"model dir           : {model_dir}<br>"
        f"pipeline stages     : {_format_pipeline_stages(classifier_model)}<br>"
        f"hash family         : MurmurHash3_x86_32 (HashingTF, seed=42)<br>"
        f"serve == train      : yes (no portable roundtrip)"
        "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "this tab uses apache spark mllib end-to-end for inference. hash "
        "buckets are exact-match with training (spark MurmurHash3, seed=42). "
        "the standard Triage Bot tab uses numpy with mmh3 for sub-second "
        "response on streamlit cloud's no-java environment."
    )
