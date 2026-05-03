"""
pipeline status - artifact checklist. shows which phase outputs are
currently in the repo, grouped by phase, with a progress bar at the top.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import all_artifacts_present, load_json
from dashboard.charts import PURPLE_PRIMARY, PURPLE_LIGHT, GRAY_TEXT, GRAY_MUTED


_ARTIFACTS = [
    ('Phase 1', 'class distribution', 'phase_1_class_dist_png', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 2', 'preprocess stats', 'phase_2_preprocess_stats', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 3', 'classifier portable', 'phase_3_classifier_portable', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 3', 'classifier summary', 'phase_3_classifier_summary', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 3', 'per-class metrics', 'phase_3_per_class_metrics', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 3', 'confusion matrix', 'phase_3_confusion_matrix', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 4', 'regressor portable', 'phase_4_regressor_portable', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 4', 'regressor summary', 'phase_4_regressor_summary', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 4', 'per-category MAE', 'phase_4_per_category_mae', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 5', 'Word2Vec KeyedVectors', 'phase_5_word2vec_kv', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 5', 'cluster summary', 'phase_5_cluster_summary', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 5', 'silhouette curve', 'phase_5_silhouette_curve', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 6', 'geo summary', 'phase_6_geo_summary', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 6', 'borough volume', 'phase_6_borough_volume', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 6', 'borough fingerprints', 'phase_6_borough_fingerprints', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 6', 'boroughs geojson', 'phase_6_boroughs_geojson', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 10', 'BERT cluster summary', 'phase_10_bert_summary', 'FINAL_PROJECT_311_NLP.ipynb'),
    ('Phase 10', 'BERT vs W2V plot', 'phase_10_bert_vs_w2v_plot', 'FINAL_PROJECT_311_NLP.ipynb'),
]


def render() -> None:
    st.header('Pipeline Status')
    st.markdown(
        '<p style="font-size: 15px; color: #4A4A4A; line-height: 1.55; max-width: 800px;">'
        'What artifacts are currently published to the repo. Each phase '
        'notebook has an auto-commit cell at the end that pushes its outputs; '
        'running the notebook is enough to light up its row here.'
        '</p>',
        unsafe_allow_html=True,
    )

    presence = all_artifacts_present()
    n_total = len(_ARTIFACTS)
    n_present = sum(1 for _, _, k, _ in _ARTIFACTS if presence.get(k, False))

    # ---- progress + headline metrics ----
    pct = n_present / n_total
    cols = st.columns(3)
    with cols[0]:
        st.metric('Artifacts published', f'{n_present} / {n_total}')
    with cols[1]:
        st.metric('Pipeline complete', f'{pct * 100:.0f}%')
    with cols[2]:
        phases_done = len({phase for phase, _, k, _ in _ARTIFACTS
                           if presence.get(k, False)})
        st.metric('Phases with output', f'{phases_done}')

    st.progress(pct)

    # ---- snapshot stats ----
    st.markdown('### Snapshot of training stats')
    snap_cols = st.columns(4)

    geo_summary = load_json('geo_summary.json')
    classifier_summary = load_json('classifier_summary.json')
    regressor_summary = load_json('regressor_summary.json')
    bert_summary = load_json('bert_cluster_summary.json')

    with snap_cols[0]:
        if geo_summary:
            st.metric('Rows aggregated', f'{geo_summary["rows_aggregated"]:,}')
        else:
            st.metric('Rows aggregated', 'pending')
    with snap_cols[1]:
        if classifier_summary:
            st.metric('Classifier macro-F1',
                      f'{classifier_summary.get("metrics", {}).get("f1", 0):.3f}')
        else:
            st.metric('Classifier macro-F1', 'pending')
    with snap_cols[2]:
        if regressor_summary:
            st.metric('Regressor lift',
                      f'{regressor_summary.get("improvement_pct", 0):.1f}%')
        else:
            st.metric('Regressor lift', 'pending')
    with snap_cols[3]:
        if bert_summary:
            st.metric('BERT silhouette',
                      f'{bert_summary.get("best_silhouette", 0):.3f}')
        else:
            st.metric('BERT silhouette', 'pending')

    # ---- per-phase artifact list ----
    st.markdown('### Artifacts by phase')

    rows_by_phase = {}
    for phase, label, key, nb in _ARTIFACTS:
        rows_by_phase.setdefault(phase, []).append((label, key, nb))

    for phase, rows in rows_by_phase.items():
        n_done = sum(1 for _, k, _ in rows if presence.get(k, False))
        n_phase = len(rows)
        complete = n_done == n_phase
        status_color = PURPLE_PRIMARY if complete else PURPLE_LIGHT
        status_text = 'complete' if complete else f'{n_done}/{n_phase}'

        # phase header
        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 12px; '
            f'margin-top: 16px; padding-bottom: 6px; '
            f'border-bottom: 1px solid #EDE8F2;">'
            f'<span style="font-size: 14px; font-weight: 600; '
            f'color: {GRAY_TEXT};">{phase}</span>'
            f'<span style="font-size: 11px; padding: 2px 8px; '
            f'background: {status_color}; color: white; border-radius: 10px; '
            f'font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">'
            f'{status_text}</span>'
            f'<span style="font-size: 12px; color: {GRAY_MUTED};">'
            f'{rows[0][2]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # phase artifact rows
        for label, key, nb in rows:
            ok = presence.get(key, False)
            icon_color = PURPLE_PRIMARY if ok else '#D5CFE0'
            icon = '&#10003;' if ok else '&middot;'
            st.markdown(
                f'<div style="display: flex; align-items: center; gap: 10px; '
                f'padding: 4px 0;">'
                f'<span style="display: inline-block; width: 18px; height: 18px; '
                f'border-radius: 50%; background: {icon_color}; '
                f'color: white; text-align: center; line-height: 18px; '
                f'font-size: 11px; font-weight: 700;">{icon}</span>'
                f'<span style="font-size: 13px; color: {GRAY_TEXT};">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
