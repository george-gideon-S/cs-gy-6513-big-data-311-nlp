"""
cluster atlas - word2vec vs bert toggle, per-cluster term lists, and
per-borough word clouds. degrades gracefully if cluster artifacts are
not yet pushed.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import load_json, render_missing


def _show_clusters(cluster_data: dict, source_label: str) -> None:
    """render a list of clusters with their top terms and dominant categories."""
    if not cluster_data:
        return

    # phase 5 cluster_summary structure: {cluster_top_terms: {0: [(term, count), ...]},
    #                                     cluster_categories: {0: [(name, cnt, pct), ...]}}
    # phase 10 bert_cluster_summary: {bert_cluster_categories: {0: [(name, cnt, pct), ...]}}
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
        st.metric(f'{source_label} best silhouette', f'{best_silhouette:.3f}',
                  help=f'at k={best_k}')

    if not cats:
        st.caption(f'{source_label} cluster category mapping not present in artifact.')
        return

    # render up to 10 clusters as expandable cards
    cluster_ids = sorted(cats.keys(), key=lambda x: int(x))
    for cid in cluster_ids[:10]:
        cluster_cats = cats[cid]
        if not cluster_cats:
            continue
        # cluster_cats is list of (name, count, pct) - top 3 per cluster
        dominant = cluster_cats[0]
        title = f'Cluster {cid}: {dominant[0]} ({dominant[2]}%)'
        with st.expander(title):
            st.caption('**Category mix:**')
            for name, cnt, pct in cluster_cats:
                st.write(f'- {name}: {pct}% ({cnt:,} complaints)')
            if cid in terms:
                top_terms = terms[cid]
                st.caption('**Top terms:**')
                st.write(', '.join(f'`{t[0]}`' for t in top_terms[:8]))


def render() -> None:
    st.header('Cluster Atlas')
    st.write(
        "Latent groupings the model finds in the descriptions, and how they "
        'compare to NYC 311\'s official taxonomy. Toggle between the Word2Vec '
        "embeddings (Phase 5) and the pretrained BERT MiniLM embeddings (Phase 10) "
        'to see how cluster quality changes.'
    )

    method = st.radio(
        'Embedding method',
        ['Word2Vec (Phase 5)', 'BERT MiniLM (Phase 10)'],
        horizontal=True,
    )

    if method.startswith('Word2Vec'):
        data = load_json('cluster_summary.json')
        if not data:
            render_missing('Phase 5', '05_word2vec.ipynb',
                           what='Word2Vec cluster summary')
        else:
            _show_clusters(data, 'Word2Vec')
    else:
        data = load_json('bert_cluster_summary.json')
        if not data:
            render_missing('Phase 10', '10_bert_embed.ipynb',
                           what='BERT cluster summary')
        else:
            _show_clusters(data, 'BERT MiniLM')

    # comparison plot if available
    st.subheader('Silhouette comparison')
    from dashboard.data_loader import asset_path
    cmp_png = asset_path('bert_vs_w2v.png')
    if cmp_png.exists():
        st.image(str(cmp_png), use_container_width=True)
    else:
        sil_png = asset_path('silhouette_curve.png')
        if sil_png.exists():
            st.image(str(sil_png), caption='Word2Vec only - run Phase 10 for the comparison',
                     use_container_width=True)
        else:
            st.caption('No silhouette curve yet - run Phases 5 and 10.')

    # ---- per-borough word clouds ----
    st.markdown('---')
    st.subheader('Per-borough complaint vocabulary')
    fingerprints = load_json('borough_fingerprints.json')
    if not fingerprints:
        render_missing('Phase 6', '06_geo_census.ipynb',
                       what='Per-borough vocabulary')
        return

    boroughs = list(fingerprints.keys())
    selected = st.selectbox('Select a borough', boroughs)

    terms = fingerprints[selected]
    # bar chart of distinctive terms by lift - works on any deployment
    # without depending on the wordcloud library which fails to compile
    # on streamlit cloud
    chart = {'term': [t[0] for t in terms], 'lift': [t[1] for t in terms]}
    st.bar_chart(chart, x='term', y='lift', height=320)
    st.caption(
        f'Each bar shows how much more common that term is in {selected} '
        'than in the corpus average. Higher lift = more distinctive of this borough.'
    )
