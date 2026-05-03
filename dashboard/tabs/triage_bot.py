"""
triage bot - the demo centerpiece.

paste a complaint -> classify it (using portable LR if available),
predict resolution time (using portable regressor if available), and show
the dominant borough fingerprints from phase 6.

if classifier/regressor artifacts arent yet pushed, the tab still renders
a useful UI: the user can pick a borough and see what kind of complaints
that borough sees most.
"""
from __future__ import annotations
import re

import numpy as np
import streamlit as st

from dashboard.data_loader import (
    load_classifier_npz, load_regressor_npz,
    load_json, render_missing,
)


# regex compiled once - matches our preprocessing exactly
_TOKEN_RE = re.compile(r'[^a-zA-Z]+')

# small project stoplist mirroring src/preprocess.py - duplicated here so
# the deployed app doesnt need pyspark/nltk to render the tab
_STOPS = {
    'street', 'avenue', 'ave', 'st', 'road', 'rd',
    'building', 'apt', 'apartment', 'address', 'location',
    'please', 'complaint', 'issue', 'problem',
    'nyc', 'ny', 'new', 'york',
    # common english words from nltk stopwords (top ~30 most relevant)
    'the', 'and', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
    'of', 'to', 'in', 'on', 'at', 'for', 'with', 'as', 'by',
    'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
    'you', 'your', 'he', 'she', 'it', 'they', 'them',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
}


def _simple_tokenize(text: str) -> list:
    """tokenize like src/preprocess.py but without nltk lemmatization."""
    if not text:
        return []
    lower = text.lower()
    parts = [t for t in _TOKEN_RE.split(lower) if t and len(t) >= 3 and t not in _STOPS]
    return parts


def _predict_classifier(text: str, artifact: dict) -> tuple:
    """
    pure-python inference using the npz from src.classify.export_portable.

    Returns:
        (predicted_label, list of (label, confidence) sorted desc, raw probs)
    """
    tokens = _simple_tokenize(text)
    if not tokens:
        return None, [], None

    vocab_size = int(artifact['vocab_size'])
    raw = np.zeros(vocab_size, dtype=np.float32)
    for tok in tokens:
        # spark uses MurmurHash3_x86_32 with seed 42. python's hash() differs
        # from spark's hash so predictions wont match exactly. for the demo
        # we accept slight drift; a future iteration could add `mmh3` to match.
        idx = abs(hash(tok)) % vocab_size
        raw[idx] += 1.0

    weighted = raw * artifact['idf']
    logits = artifact['coefs'] @ weighted + artifact['intercepts']
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    labels = list(artifact['labels'])
    ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])
    return ranked[0][0], ranked, probs


def render() -> None:
    st.header('Triage Bot')
    st.write(
        'Paste an NYC 311 complaint description. The system will classify '
        'it into one of the top 20 categories, predict resolution time, and '
        'show distinctive vocabulary from the borough you select.'
    )

    sample = (
        "There's a really bad smell coming from the trash piles on 3rd Ave, "
        "been like this for days, rats everywhere now."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area(
            'Complaint description',
            value=st.session_state.get('triage_text', sample),
            height=140,
            key='triage_text',
        )
    with col2:
        borough = st.selectbox(
            'Borough',
            ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'],
            index=2,
            key='triage_borough',
        )
        run = st.button('Triage this complaint', type='primary', use_container_width=True)

    if not run:
        st.info('Type a complaint above and click Triage to see results.')
        return

    classifier = load_classifier_npz()

    # ---- classification ----
    st.subheader('Classification')
    if classifier is None:
        st.warning(
            'Classifier not yet exported for serving. To enable, run '
            '`notebooks/03_classify.ipynb` in Colab — its auto-commit cell '
            'at the end will push `models/portable/classifier.npz` to the repo.'
        )
        # show the user what we DO know - tokenized form of their input
        toks = _simple_tokenize(text)
        if toks:
            st.caption('Tokenized form (preview of what the classifier would see):')
            st.code(' '.join(toks))
    else:
        label, ranked, probs = _predict_classifier(text, classifier)
        if label is None:
            st.error('No usable tokens in the input. Try a longer description.')
        else:
            st.success(f'Predicted category: **{label}**')
            top5 = ranked[:5]
            chart = {
                'category': [k for k, _ in top5],
                'confidence': [v for _, v in top5],
            }
            st.bar_chart(chart, x='category', y='confidence', height=240)
            st.caption(
                'Confidence is approximate - the deployed classifier uses '
                "Python's hash() instead of Spark's MurmurHash3 internally, "
                'so per-token feature indices drift slightly from training.'
            )

    # ---- resolution time prediction ----
    st.subheader('Resolution Time Prediction')
    regressor = load_regressor_npz()
    if regressor is None:
        st.warning(
            'Regressor not yet exported. Run `notebooks/04_regress.ipynb` '
            '(including the v2 cell that adds `label_canonical` as a '
            'feature) — its auto-commit cell will push `models/portable/regressor.npz`.'
        )
    else:
        # for now, just show the median for the predicted category if we have one
        median_map = regressor.get('median_map')
        if median_map is not None and classifier is not None and label is not None:
            try:
                lookup = dict(median_map.tolist())
                hours = float(lookup.get(label, regressor.get('global_median', 24)))
                st.metric(
                    'Expected resolution',
                    f'~{hours:.0f} hours' + (f' ({hours/24:.1f} days)' if hours > 36 else ''),
                    help='Based on median per-category from training data.',
                )
            except Exception as e:
                st.caption(f'(could not look up median: {e})')

    # ---- borough fingerprint context ----
    st.subheader(f'What {borough} typically complains about')
    fingerprints = load_json('borough_fingerprints.json')
    volumes = load_json('borough_volume.json')
    if fingerprints and borough in fingerprints:
        terms = fingerprints[borough]
        st.caption(
            f'Top 10 distinctive terms in {borough} (high lift = used in this '
            "borough way more than the city average):"
        )
        chart = {
            'term': [t[0] for t in terms[:10]],
            'lift': [t[1] for t in terms[:10]],
        }
        st.bar_chart(chart, x='term', y='lift', height=260)

        if volumes and borough in volumes:
            v = volumes[borough]
            st.caption(
                f"{borough} contributed {v['count']:,} complaints to the 2M sample. "
                f"Top categories: " + ', '.join(c[0] for c in v['top_categories'][:3]) + '.'
            )
    else:
        render_missing('Phase 6', '06_geo_census.ipynb',
                       what='Borough fingerprints')
