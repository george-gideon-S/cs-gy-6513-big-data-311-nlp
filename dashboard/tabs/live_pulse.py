"""
live pulse - the streaming PoC tab. since the deployed app cant run
spark structured streaming, this tab lets the user fetch a few recent
complaints from NYC SODA api and run them through the classifier in
real time. demonstrates the streaming pattern without the infra.
"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header('Live Pulse')
    st.write(
        "A live pull from NYC's open data API. Fetches the most recent 311 "
        'complaints right now, classifies each one, and shows the result. '
        'This is the practical version of the streaming PoC the proposal '
        'planned - we replaced Spark Structured Streaming with on-demand '
        'API pulls because the deployed app has no Spark.'
    )

    n = st.slider('How many recent complaints to fetch', 5, 100, 20)

    if not st.button('Fetch + classify recent complaints', type='primary'):
        st.info('Click the button to pull live data from NYC Open Data.')
        return

    with st.spinner('Fetching from NYC SODA API...'):
        try:
            import requests
            url = (
                'https://data.cityofnewyork.us/resource/erm2-nwe9.csv'
                f'?$limit={n}&$order=created_date DESC'
            )
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            import pandas as pd
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
        except Exception as e:
            st.error(f'SODA fetch failed: {e}')
            return

    st.success(f'Fetched {len(df):,} complaints')

    # if classifier portable is available, classify each one
    from dashboard.data_loader import load_classifier_npz
    from dashboard.tabs.triage_bot import _predict_classifier

    classifier = load_classifier_npz()

    if classifier is None:
        st.warning(
            'Classifier not yet exported. Showing raw API rows only. To '
            'enable live classification, run `notebooks/03_classify.ipynb` '
            'in Colab.'
        )
        st.dataframe(
            df[['unique_key', 'created_date', 'complaint_type', 'descriptor', 'borough']]
            .head(n),
            use_container_width=True, hide_index=True,
        )
        return

    # classify each row's descriptor
    text_col = 'descriptor' if 'descriptor' in df.columns else 'complaint_type'
    df['predicted_category'] = df[text_col].fillna('').apply(
        lambda t: _predict_classifier(t, classifier)[0] or 'Other'
    )
    df['actual_category'] = df.get('complaint_type', df.get('problem', ''))
    agreement = (df['predicted_category'] == df['actual_category']).mean()
    st.metric('Live agreement with API category', f'{agreement * 100:.0f}%')

    show_cols = ['created_date', 'borough', 'descriptor', 'actual_category', 'predicted_category']
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols].head(n), use_container_width=True, hide_index=True)
