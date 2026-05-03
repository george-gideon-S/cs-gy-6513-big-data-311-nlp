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


# typography + spacing system. one place to tune the whole app.
_GLOBAL_CSS = """
<style>
/* base type stack - inter on apple devices, system fallback */
.stApp, [data-testid="stSidebar"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* hide streamlit's default chrome that looks unprofessional */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] { gap: 4px; }

/* dashboard hero title */
.dash-hero {
    color: #57068C;
    font-size: 44px;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}
.dash-hero-sub {
    color: #6B6B6B;
    font-size: 16px;
    font-weight: 400;
    margin: 0 0 24px 0;
    letter-spacing: 0.01em;
}

/* h2 in main content - tab section headers */
.stApp h2 {
    color: #2C2C2C !important;
    font-size: 28px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    margin-top: 8px !important;
    margin-bottom: 4px !important;
}

/* h3 - subsections within a tab */
.stApp h3 {
    color: #57068C !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-top: 24px !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* tab labels - clean, no emoji, slight letter spacing */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #E5E5E5;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
    color: #6B6B6B !important;
    letter-spacing: 0.01em;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #57068C !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #57068C !important;
    height: 2px !important;
}

/* metric cards - bigger purple values, smaller uppercase labels */
[data-testid="stMetricValue"] {
    color: #57068C !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricLabel"] {
    color: #6B6B6B !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* body paragraphs */
.stApp p, .stApp li {
    color: #4A4A4A;
    font-size: 14px;
    line-height: 1.6;
}

/* sidebar polish */
[data-testid="stSidebar"] {
    background: #FAF9FC;
    border-right: 1px solid #EDE8F2;
}
[data-testid="stSidebar"] h2 {
    color: #57068C !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
    font-size: 13px !important;
    color: #4A4A4A !important;
    line-height: 1.55 !important;
}

/* button polish */
button[kind="primary"] {
    background-color: #57068C !important;
    border-color: #57068C !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 10px 20px !important;
}
button[kind="primary"]:hover {
    background-color: #3A0560 !important;
    border-color: #3A0560 !important;
}

/* radios / selectbox - tidier */
[data-baseweb="radio"] label {
    font-size: 14px !important;
}

/* dataframe - lighter borders */
[data-testid="stDataFrame"] {
    border: 1px solid #EDE8F2;
    border-radius: 4px;
}

/* keep code spans readable */
code {
    background: #F5F2F8 !important;
    color: #57068C !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
    font-size: 12px !important;
}

/* expander headers */
[data-testid="stExpander"] details summary {
    font-size: 14px !important;
    font-weight: 500 !important;
}
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title='The Language of Complaints',
        page_icon=None,
        layout='wide',
        initial_sidebar_state='expanded',
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    # hero title
    st.markdown(
        '<div class="dash-hero">The Language of Complaints</div>'
        '<div class="dash-hero-sub">'
        'NLP-powered triage on NYC 311 service requests &nbsp;&middot;&nbsp; '
        'CSGY-6513 Big Data Final Project'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('## About this project')
        st.markdown(
            'A 2 million row stratified sample of NYC 311 complaints '
            '(2010 to present), distilled into:'
        )
        st.markdown(
            '- A classifier that maps free text to one of 20 official categories\n'
            '- A regressor that estimates resolution time\n'
            '- Latent-issue clusters from Word2Vec and pretrained BERT '
            'sentence embeddings\n'
            '- Per-borough complaint vocabulary fingerprints'
        )
        st.markdown('## Team')
        st.markdown(
            'George Gideon Sale (gs4602)<br>'
            'Aayush Prranav Chandrashekar (ac11929)<br>'
            'Shreeram Sankar (ss18731)',
            unsafe_allow_html=True,
        )
        st.markdown('## Source')
        st.markdown(
            '[github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp]'
            '(https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp)'
        )

    # tabs without emojis - clean text only
    tab_names = [
        'Triage Bot',
        'City Pulse',
        'Cluster Atlas',
        'Bias Audit',
        'Live Pulse',
        'Pipeline Status',
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
