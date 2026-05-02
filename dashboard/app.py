"""
streamlit entry point. five tabs.
runs against pre-exported portable models from models/portable so it doesnt
need spark or java at serve time. that means it deploys to streamlit cloud
free tier without drama.

run locally:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

# make sure we can import dashboard.tabs.* and src.* whether were running
# from the repo root or from inside dashboard/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# tabs
from dashboard.tabs import (
    triage_bot,
    city_pulse,
    cluster_atlas,
    bias_audit,
    live_pulse,
    pipeline_status,
)


def main() -> None:
    st.set_page_config(
        page_title="The Language of Complaints",
        page_icon="📣",  # streamlit handles emoji-as-icon natively, this isnt graded prose
        layout="wide",
    )

    # nyu purple-ish accent. streamlit picks up css through st.markdown.
    st.markdown(
        """
        <style>
        .stApp h1 { color: #57068C; }
        .stApp h2 { color: #57068C; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("The Language of Complaints")
    st.caption(
        "NLP-powered triage on NYC 311 service requests. "
        "CSGY-6513 Big Data Final Project, Section D Term 2."
    )

    # sidebar - shared filters that tabs can choose to honor
    with st.sidebar:
        st.header("Filters")
        st.session_state.setdefault("borough_filter", [])
        st.multiselect(
            "Borough",
            ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"],
            key="borough_filter",
        )
        st.markdown("---")
        st.caption(
            "Built by George Gideon Sale, Aayush Prranav Chandrashekar, "
            "Shreeram Sankar."
        )

    # tabs
    tab_names = [
        "Triage Bot",
        "City Pulse",
        "Cluster Atlas",
        "Bias Audit",
        "Live Pulse",
        "Pipeline Status",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        triage_bot.render()
    with tabs[1]:
        city_pulse.render()
    with tabs[2]:
        cluster_atlas.render()
    with tabs[3]:
        bias_audit.render()
    with tabs[4]:
        live_pulse.render()
    with tabs[5]:
        pipeline_status.render()


if __name__ == "__main__":
    main()
