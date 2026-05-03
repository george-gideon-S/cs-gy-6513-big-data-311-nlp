"""
streamlit entry point for the deployed dashboard.

runs against pre-exported portable models from models/portable/ and JSON
artifacts from dashboard/assets/. no spark or java required at serve time
so this deploys cleanly to streamlit community cloud.

run locally:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

# make sure dashboard.tabs.* and src.* are importable regardless of cwd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        page_title='The Language of Complaints',
        page_icon='📣',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    # nyu purple-ish accent
    st.markdown(
        """
        <style>
        .stApp h1 { color: #57068C; }
        .stApp h2 { color: #57068C; }
        .stApp h3 { color: #6f1faf; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title('The Language of Complaints')
    st.caption(
        'NLP-powered triage on NYC 311 service requests. '
        'CSGY-6513 Big Data Final Project, Section D Term 2.'
    )

    with st.sidebar:
        st.header('About')
        st.markdown(
            'A 2 million row stratified sample of NYC 311 complaints '
            '(2010-present), distilled into:'
        )
        st.markdown(
            '- A classifier that maps free text to one of 20 official categories\n'
            '- A regressor that estimates resolution time\n'
            '- Latent-issue clusters from Word2Vec **and** pretrained BERT embeddings\n'
            '- Per-borough complaint vocabulary fingerprints'
        )
        st.markdown('---')
        st.caption(
            '**Team:** George Gideon Sale (gs4602), '
            'Aayush Prranav Chandrashekar (ac11929), '
            'Shreeram Sankar (ss18731).'
        )
        st.caption(
            'Source: '
            '[github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp]'
            '(https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp)'
        )

    tab_names = [
        '🩺 Triage Bot',
        '🗺️ City Pulse',
        '🔬 Cluster Atlas',
        '⚖️ Bias Audit',
        '📡 Live Pulse',
        '📋 Pipeline Status',
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


if __name__ == '__main__':
    main()
