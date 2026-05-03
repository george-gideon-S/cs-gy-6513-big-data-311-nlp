"""
pipeline status - one place to see what's done and what's still needed.
auto-detects each artifact in the repo and shows a green check or a
hollow circle.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import all_artifacts_present, load_json


# label and source-notebook mapping for each artifact
_ARTIFACTS = [
    ('Phase 1 - class distribution', 'phase_1_class_dist_png', '01_ingest.ipynb'),
    ('Phase 2 - preprocess stats', 'phase_2_preprocess_stats', '02_preprocess.ipynb'),
    ('Phase 3 - classifier portable .npz', 'phase_3_classifier_portable', '03_classify.ipynb'),
    ('Phase 3 - classifier summary', 'phase_3_classifier_summary', '03_classify.ipynb'),
    ('Phase 3 - per-class metrics', 'phase_3_per_class_metrics', '03_classify.ipynb'),
    ('Phase 3 - confusion matrix', 'phase_3_confusion_matrix', '03_classify.ipynb'),
    ('Phase 4 - regressor portable .npz', 'phase_4_regressor_portable', '04_regress.ipynb'),
    ('Phase 4 - regressor summary', 'phase_4_regressor_summary', '04_regress.ipynb'),
    ('Phase 4 - per-category MAE', 'phase_4_per_category_mae', '04_regress.ipynb'),
    ('Phase 5 - Word2Vec KeyedVectors', 'phase_5_word2vec_kv', '05_word2vec.ipynb'),
    ('Phase 5 - cluster summary', 'phase_5_cluster_summary', '05_word2vec.ipynb'),
    ('Phase 5 - silhouette curve', 'phase_5_silhouette_curve', '05_word2vec.ipynb'),
    ('Phase 6 - geo summary', 'phase_6_geo_summary', '06_geo_census.ipynb'),
    ('Phase 6 - borough volume', 'phase_6_borough_volume', '06_geo_census.ipynb'),
    ('Phase 6 - borough fingerprints', 'phase_6_borough_fingerprints', '06_geo_census.ipynb'),
    ('Phase 6 - boroughs geojson', 'phase_6_boroughs_geojson', '06_geo_census.ipynb'),
    ('Phase 10 - BERT cluster summary', 'phase_10_bert_summary', '10_bert_embed.ipynb'),
    ('Phase 10 - BERT vs W2V plot', 'phase_10_bert_vs_w2v_plot', '10_bert_embed.ipynb'),
]


def render() -> None:
    st.header('Pipeline Status')
    st.write(
        "What artifacts are currently published to the repo. Each phase "
        'notebook has an auto-commit cell at the end that pushes its '
        'artifacts; running the notebook is enough to light up its row here.'
    )

    presence = all_artifacts_present()
    n_total = len(_ARTIFACTS)
    n_present = sum(1 for _, k, _ in _ARTIFACTS if presence.get(k, False))

    progress_pct = n_present / n_total
    st.progress(progress_pct,
                text=f'{n_present}/{n_total} artifacts published ({progress_pct*100:.0f}%)')

    # group by phase
    rows_by_phase = {}
    for label, key, nb in _ARTIFACTS:
        phase = label.split(' - ')[0]
        rows_by_phase.setdefault(phase, []).append((label, key, nb))

    for phase, rows in rows_by_phase.items():
        with st.expander(phase, expanded=any(presence.get(k, False) for _, k, _ in rows)):
            for label, key, nb in rows:
                ok = presence.get(key, False)
                icon = '✅' if ok else '⏳'
                st.markdown(f'{icon} **{label.split(" - ", 1)[-1]}** — `notebooks/{nb}`')

    # ---- summary stats from artifacts that ARE present ----
    st.markdown('---')
    st.subheader('Snapshot of training stats')

    snap_cols = st.columns(4)

    geo_summary = load_json('geo_summary.json')
    if geo_summary:
        with snap_cols[0]:
            st.metric('Rows aggregated (Phase 6)', f'{geo_summary["rows_aggregated"]:,}')

    classifier_summary = load_json('classifier_summary.json')
    if classifier_summary:
        with snap_cols[1]:
            st.metric('Classifier macro-F1', f'{classifier_summary.get("metrics", {}).get("f1", 0):.3f}')

    regressor_summary = load_json('regressor_summary.json')
    if regressor_summary:
        with snap_cols[2]:
            st.metric('Regressor lift over baseline',
                      f'{regressor_summary.get("improvement_pct", 0):.1f}%')

    bert_summary = load_json('bert_cluster_summary.json')
    if bert_summary:
        with snap_cols[3]:
            st.metric('BERT silhouette',
                      f'{bert_summary.get("best_silhouette", 0):.3f}')
