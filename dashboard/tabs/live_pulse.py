"""
live pulse - on-demand fetch from NYC SODA api + classify each row.
replaces the proposal's Spark Structured Streaming PoC because the
deployed app has no Spark.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import load_classifier_npz
from dashboard.charts import empty_state, GRAY_MUTED, GRAY_TEXT


def render() -> None:
    st.header('Live Pulse')
    st.markdown(
        '<p style="font-size: 15px; color: #4A4A4A; line-height: 1.55; max-width: 800px;">'
        "A live pull from NYC's open data API. Fetches the most recent 311 "
        'complaints right now, classifies each one, and shows the result. '
        'This is the practical version of the streaming proof-of-concept the '
        'proposal planned - we replaced Spark Structured Streaming with on-demand '
        'API pulls because the deployed app has no Spark runtime.'
        '</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        n = st.slider('How many recent complaints to fetch', 5, 100, 25, key='live_n')
    with col2:
        st.write('')
        run = st.button('Fetch + classify', type='primary', use_container_width=True)

    if not run:
        st.markdown(
            '<div style="margin-top: 14px; padding: 14px 18px; '
            'background: #F5F2F8; border-left: 3px solid #9B6FC2; '
            'border-radius: 4px; font-size: 14px; color: #4A4A4A;">'
            'Click <b style="color: #57068C;">Fetch + classify</b> to pull '
            'live data from NYC Open Data and run our classifier on it.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    with st.spinner('Fetching from NYC SODA API...'):
        try:
            import requests
            from io import StringIO
            import pandas as pd

            url = (
                'https://data.cityofnewyork.us/resource/erm2-nwe9.csv'
                f'?$limit={n}&$order=created_date DESC'
            )
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
        except Exception as e:
            st.error(f'SODA fetch failed: {e}')
            return

    # ---- live metrics ----
    classifier = load_classifier_npz()
    text_col = 'descriptor' if 'descriptor' in df.columns else 'complaint_type'
    if classifier is not None and text_col in df.columns:
        from dashboard.tabs.triage_bot import _predict
        df['predicted_category'] = df[text_col].fillna('').apply(
            lambda t: _predict(t, classifier)[0] or 'Other'
        )
        df['actual_category'] = df.get('complaint_type', df.get('problem', ''))
        agreement = (df['predicted_category'] == df['actual_category']).mean()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric('Fetched', f'{len(df):,}')
        with m2:
            st.metric('Live agreement with API', f'{agreement * 100:.0f}%')
        with m3:
            from datetime import datetime
            st.metric('Most recent', str(df['created_date'].max())[:16] if 'created_date' in df.columns else 'n/a')
    else:
        if classifier is None:
            empty_state(
                'Showing raw API rows only. Live classification needs the '
                'portable classifier - run '
                '<code>notebooks/FINAL_PROJECT_311_NLP.ipynb</code>.',
                action='models/portable/classifier.npz',
            )
        m1 = st.columns(1)[0]
        with m1:
            st.metric('Fetched', f'{len(df):,}')

    # ---- table ----
    st.markdown('### Recent complaints')
    show_cols = ['created_date', 'borough', 'descriptor',
                 'actual_category', 'predicted_category']
    show_cols = [c for c in show_cols if c in df.columns]
    if not show_cols:
        show_cols = list(df.columns)[:5]
    st.dataframe(df[show_cols].head(n), use_container_width=True, hide_index=True)
