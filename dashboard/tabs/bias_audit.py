"""
bias audit - per-class metrics + headline numbers, anchored to
kontokosta and hong (2021) on socio-spatial 311 reporting biases.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import load_json, asset_path
from dashboard.charts import empty_state, GRAY_TEXT, GRAY_MUTED


def render() -> None:
    st.header('Bias Audit')
    st.markdown(
        '<p style="font-size: 15px; color: #4A4A4A; line-height: 1.55; max-width: 800px;">'
        'Models trained on 311 inherit reporting biases from the data itself: '
        'who calls 311, what they call about, and how operators transcribe '
        'their words. Following Kontokosta &amp; Hong (2021, '
        '<i>Sustainable Cities and Society</i>), this tab surfaces where our '
        'classifier performs unevenly across categories - a check on where the '
        'underlying reporting itself is uneven.'
        '</p>',
        unsafe_allow_html=True,
    )

    summary = load_json('classifier_summary.json')
    per_class = load_json('per_class_metrics.json')

    if not summary or not per_class:
        empty_state(
            'Per-class metrics not yet published. Run '
            '<code>notebooks/FINAL_PROJECT_311_NLP.ipynb</code> in Colab and the '
            'auto-commit cell will push these artifacts.',
            action='dashboard/assets/classifier_summary.json + per_class_metrics.json',
        )
        return

    # ---- headline metrics ----
    metrics = summary.get('metrics', {})
    baselines = summary.get('baselines', {})
    cols = st.columns(4)
    with cols[0]:
        st.metric('Macro-F1', f'{metrics.get("f1", 0):.3f}')
    with cols[1]:
        st.metric('Accuracy', f'{metrics.get("accuracy", 0):.3f}')
    with cols[2]:
        st.metric('Test rows', f'{summary.get("n_test", 0):,}')
    with cols[3]:
        lift_kw = summary.get('lift_over_keyword_f1', 0)
        st.metric('Lift over keyword baseline', f'+{lift_kw:.3f}')

    # ---- baseline comparison ----
    st.markdown('### Baselines required by the proposal')
    if baselines:
        import pandas as pd
        b = pd.DataFrame([
            {'model': 'Majority class',
             'macro F1': baselines.get('majority_class', {}).get('macro_f1', 0),
             'accuracy': baselines.get('majority_class', {}).get('accuracy', 0)},
            {'model': 'Keyword heuristic',
             'macro F1': baselines.get('keyword_heuristic', {}).get('macro_f1', 0),
             'accuracy': baselines.get('keyword_heuristic', {}).get('accuracy', 0)},
            {'model': 'TF-IDF + Logistic Regression (ours)',
             'macro F1': metrics.get('f1', 0),
             'accuracy': metrics.get('accuracy', 0)},
        ])
        st.dataframe(
            b.style.format({'macro F1': '{:.3f}', 'accuracy': '{:.3f}'}),
            use_container_width=True, hide_index=True,
        )
        st.markdown(
            f'<p style="font-size: 13px; color: {GRAY_MUTED}; margin-top: -8px;">'
            'The keyword baseline already hits 0.78 - that tells us 311 '
            'descriptors are mostly drop-down dictionary terms rather than '
            "free text. LR's discriminative weighting still adds another 18 "
            'percentage points. Honest framing for the demo: the model is '
            "not doing magic NLP, it is learning a lookup with better "
            'discrimination on the harder classes.'
            '</p>',
            unsafe_allow_html=True,
        )

    # ---- per-class table ----
    st.markdown('### Per-class precision, recall, F1')
    st.markdown(
        f'<p style="font-size: 13px; color: {GRAY_MUTED};">'
        'Watch for low-F1 classes - those are where the model makes the '
        'biggest mistakes. From training, '
        '<b style="color: #57068C;">Noise - Street/Sidewalk</b> had F1 ~0.37 '
        'because its descriptors heavily overlap with '
        '<b style="color: #57068C;">Noise - Residential</b>.'
        '</p>',
        unsafe_allow_html=True,
    )
    import pandas as pd
    df = pd.DataFrame(per_class).sort_values('support', ascending=False)
    st.dataframe(
        df.style.format({
            'precision': '{:.3f}', 'recall': '{:.3f}',
            'f1': '{:.3f}', 'support': '{:,}',
        }),
        use_container_width=True, hide_index=True,
    )

    # ---- confusion matrix ----
    cm = asset_path('cm.png')
    if cm.exists():
        st.markdown('### Confusion matrix')
        st.image(str(cm),
                 caption=f'Test set normalized confusion matrix - macro-F1 = '
                         f'{metrics.get("f1", 0):.3f}')
