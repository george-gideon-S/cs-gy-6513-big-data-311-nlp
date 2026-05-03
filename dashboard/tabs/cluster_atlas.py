"""
cluster atlas - word2vec vs bert toggle, per-cluster cards, per-borough
fingerprint bar chart.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import load_json, asset_path
from dashboard.charts import (
    horizontal_bar, empty_state, PURPLE_PRIMARY, PURPLE_LIGHT, PURPLE_DEEP,
    GRAY_TEXT, GRAY_MUTED,
)


def _show_clusters(cluster_data: dict, source_label: str) -> None:
    if source_label.startswith('Word2Vec'):
        terms = cluster_data.get('cluster_top_terms', {})
        cats = cluster_data.get('cluster_categories', {})
        sweep = cluster_data.get('sweep', [])
    else:
        terms = {}
        cats = cluster_data.get('bert_cluster_categories', {})
        sweep = cluster_data.get('kmeans_sweep', [])

    best_silhouette = max((s.get('silhouette', 0) for s in sweep), default=0)
    best_k = cluster_data.get('best_k') or cluster_data.get('kmeans', {}).get('best_k')

    if best_k is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Embedding source', source_label)
        with col2:
            st.metric('Best k (silhouette)', f'k = {best_k}')
        with col3:
            st.metric('Silhouette score', f'{best_silhouette:.3f}')

    if not cats:
        st.caption(f'{source_label} cluster category mapping not present in artifact.')
        return

    st.markdown('### Cluster -> official-category breakdown')
    st.markdown(
        '<p style="font-size: 13px; color: #6B6B6B; margin-bottom: 12px;">'
        'Each cluster is an emergent grouping the embeddings discovered. '
        'The percentages show what fraction of the cluster maps to each '
        "of NYC's official categories. A 100% cluster aligns perfectly with "
        'the taxonomy; a mixed cluster reveals latent cross-category structure.'
        '</p>',
        unsafe_allow_html=True,
    )

    cluster_ids = sorted(cats.keys(), key=lambda x: int(x))
    # render as a 2-column grid of cards
    cols_per_row = 2
    for i in range(0, len(cluster_ids), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, cid in zip(cols, cluster_ids[i:i + cols_per_row]):
            cluster_cats = cats[cid]
            if not cluster_cats:
                continue
            with col:
                dominant = cluster_cats[0]
                # build category mix bars
                bars_html = ''
                for name, cnt, pct in cluster_cats:
                    width = max(2, int(pct))
                    bars_html += (
                        f'<div style="margin: 6px 0;">'
                        f'<div style="display: flex; justify-content: space-between; '
                        f'font-size: 12px; color: {GRAY_TEXT}; margin-bottom: 2px;">'
                        f'<span style="overflow: hidden; text-overflow: ellipsis; '
                        f'white-space: nowrap; margin-right: 8px;">{name}</span>'
                        f'<span style="color: {GRAY_MUTED}; font-variant-numeric: tabular-nums;">'
                        f'{pct}%</span>'
                        f'</div>'
                        f'<div style="background: #F0EAF7; height: 4px; border-radius: 2px;">'
                        f'<div style="width: {width}%; height: 100%; '
                        f'background: {PURPLE_PRIMARY}; border-radius: 2px;"></div>'
                        f'</div></div>'
                    )

                terms_html = ''
                if cid in terms and terms[cid]:
                    chips = ''.join(
                        f'<span style="background: #F2EFF7; color: {PURPLE_DEEP}; '
                        f'padding: 2px 8px; border-radius: 10px; font-size: 11px; '
                        f'margin: 2px; display: inline-block; '
                        f'font-family: Menlo, monospace;">{t[0]}</span>'
                        for t in terms[cid][:6]
                    )
                    terms_html = (
                        f'<div style="margin-top: 10px; padding-top: 10px; '
                        f'border-top: 1px solid #F0F0F0;">'
                        f'<div style="font-size: 11px; color: {GRAY_MUTED}; '
                        f'text-transform: uppercase; letter-spacing: 0.06em; '
                        f'margin-bottom: 4px;">Top terms</div>'
                        f'<div>{chips}</div></div>'
                    )

                st.markdown(
                    f'<div style="background: white; padding: 14px 16px; '
                    f'border-radius: 6px; border: 1px solid #EDE8F2; '
                    f'margin-bottom: 8px;">'
                    f'<div style="display: flex; justify-content: space-between; '
                    f'align-items: baseline; margin-bottom: 8px;">'
                    f'<div style="font-size: 11px; color: {GRAY_MUTED}; '
                    f'text-transform: uppercase; letter-spacing: 0.06em;">'
                    f'Cluster {cid}</div>'
                    f'<div style="font-size: 12px; color: {PURPLE_PRIMARY}; '
                    f'font-weight: 600;">{dominant[0]}</div>'
                    f'</div>'
                    f'{bars_html}'
                    f'{terms_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render() -> None:
    st.header('Cluster Atlas')
    st.markdown(
        '<p style="font-size: 15px; color: #4A4A4A; line-height: 1.55; max-width: 800px;">'
        'Latent groupings the model discovers in the descriptions, and how '
        "they compare to NYC 311's official taxonomy. Toggle between the "
        'Word2Vec embeddings (Phase 5) and the pretrained BERT MiniLM '
        'embeddings (Phase 10) to see how cluster quality changes.'
        '</p>',
        unsafe_allow_html=True,
    )

    method = st.radio(
        'Embedding method',
        ['Word2Vec (Phase 5)', 'BERT MiniLM (Phase 10)'],
        horizontal=True,
        label_visibility='collapsed',
    )

    if method.startswith('Word2Vec'):
        data = load_json('cluster_summary.json')
        if not data:
            empty_state(
                'Word2Vec cluster summary not yet published. Run '
                '<code>notebooks/05_word2vec.ipynb</code> in Colab.',
                action='dashboard/assets/cluster_summary.json',
            )
        else:
            _show_clusters(data, 'Word2Vec')
    else:
        data = load_json('bert_cluster_summary.json')
        if not data:
            empty_state(
                'BERT cluster summary not yet published. Run '
                '<code>notebooks/10_bert_embed.ipynb</code> in Colab on a GPU runtime.',
                action='dashboard/assets/bert_cluster_summary.json',
            )
        else:
            _show_clusters(data, 'BERT MiniLM')

    # ---- silhouette comparison plot ----
    st.markdown('### Silhouette comparison')
    cmp_png = asset_path('bert_vs_w2v.png')
    sil_png = asset_path('silhouette_curve.png')
    if cmp_png.exists():
        st.image(str(cmp_png), use_container_width=True)
    elif sil_png.exists():
        st.image(str(sil_png),
                 caption='Word2Vec only - run Phase 10 for the BERT comparison',
                 use_container_width=True)
    else:
        empty_state(
            'Silhouette curve not yet generated. Run Phase 5 (Word2Vec) and '
            'Phase 10 (BERT) to populate this comparison.',
        )

    # ---- per-borough vocabulary fingerprint ----
    st.markdown('### Per-borough complaint vocabulary')
    fingerprints = load_json('borough_fingerprints.json')
    if not fingerprints:
        empty_state(
            'Per-borough vocabulary not yet published. Run '
            '<code>notebooks/06_geo_census.ipynb</code> in Colab.',
            action='dashboard/assets/borough_fingerprints.json',
        )
        return

    boroughs = list(fingerprints.keys())
    selected = st.selectbox('Select a borough', boroughs)

    terms = fingerprints[selected]
    fig = horizontal_bar(
        labels=[t[0] for t in terms[:15]],
        values=[t[1] for t in terms[:15]],
        height=440,
        x_label='lift (vs corpus average)',
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f'Each bar shows how much more common the term is in {selected} '
        f'than in the city as a whole. Higher lift = more distinctive.'
    )
