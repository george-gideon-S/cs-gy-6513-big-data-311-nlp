"""
bias audit - per-class metrics from phase 3, plus a discussion anchored
to kontokosta and hong (2021) on socio-spatial 311 reporting biases.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import load_json, asset_path, render_missing


def render() -> None:
    st.header('Bias Audit')
    st.write(
        'Models trained on 311 inherit reporting biases from the data: who '
        'calls 311, what they call about, and how operators transcribe '
        'their words. Following Kontokosta & Hong (2021, *Sustainable '
        'Cities and Society*), this tab surfaces where our classifier '
        'performs unevenly across categories - which doubles as a check on '
        'where the underlying reporting itself is uneven.'
    )

    summary = load_json('classifier_summary.json')
    per_class = load_json('per_class_metrics.json')

    if not summary or not per_class:
        render_missing('Phase 3', '03_classify.ipynb',
                       what='Per-class metrics')
        return

    # ---- headline metrics ----
    metrics = summary.get('metrics', {})
    baselines = summary.get('baselines', {})
    cols = st.columns(3)
    with cols[0]:
        st.metric('Macro-F1', f'{metrics.get("f1", 0):.3f}',
                  delta=f'+{summary.get("lift_over_keyword_f1", 0):.3f} vs keyword')
    with cols[1]:
        st.metric('Accuracy', f'{metrics.get("accuracy", 0):.3f}')
    with cols[2]:
        st.metric('Test set size', f'{summary.get("n_test", 0):,}')

    # ---- per-class table ----
    st.subheader('Per-class precision, recall, F1')
    st.write(
        "Watch for low-F1 classes - those are where the model makes the "
        "biggest mistakes. From training, **Noise - Street/Sidewalk** "
        "had F1 ~0.37 because its descriptors heavily overlap with "
        '**Noise - Residential**.'
    )
    import pandas as pd
    df = pd.DataFrame(per_class).sort_values('support', ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- confusion matrix ----
    cm = asset_path('cm.png')
    if cm.exists():
        st.subheader('Confusion matrix')
        st.image(str(cm), caption=f'Test set normalized confusion matrix - macro-F1 = {metrics.get("f1", 0):.3f}')

    # ---- baseline comparison ----
    st.subheader('Baselines required by the proposal')
    if baselines:
        b = pd.DataFrame([
            {'model': 'Majority class', 'macro_f1': baselines.get('majority_class', {}).get('macro_f1', 0),
             'accuracy': baselines.get('majority_class', {}).get('accuracy', 0)},
            {'model': 'Keyword heuristic', 'macro_f1': baselines.get('keyword_heuristic', {}).get('macro_f1', 0),
             'accuracy': baselines.get('keyword_heuristic', {}).get('accuracy', 0)},
            {'model': 'TF-IDF + Logistic Regression (ours)', 'macro_f1': metrics.get('f1', 0),
             'accuracy': metrics.get('accuracy', 0)},
        ])
        st.dataframe(b, use_container_width=True, hide_index=True)
        st.caption(
            "The keyword baseline already hits 0.78 - that tells us "
            '311 descriptors are mostly drop-down dictionary terms rather '
            "than free text. LR's discriminative weighting still adds "
            'another 18 percentage points. Honest framing for the demo: '
            "the model isn't doing magic NLP, it's learning a lookup with "
            'better discrimination on rare classes.'
        )
