"""
triage bot - the demo centerpiece.

paste a complaint -> classify it (using portable LR if available),
predict resolution time (using portable regressor if available), and show
distinctive vocabulary from the borough. when artifacts are missing the
tab still presents useful borough context so the user always sees value.
"""
from __future__ import annotations
import re

import numpy as np
import streamlit as st

# spark-compatible token hash. spark's MLlib HashingTF uses MurmurHash3_x86_32
# with seed=42 internally, so we replicate that here. python's built-in hash()
# is randomized per process (PYTHONHASHSEED) so it is NOT compatible with
# the buckets the trained classifier learned. using mmh3 with the same seed
# makes deploy-time tokens land in the same buckets training-time tokens did.
try:
    import mmh3
    _HAVE_MMH3 = True
except ImportError:
    _HAVE_MMH3 = False


def _bucket(token: str, vocab_size: int) -> int:
    """spark-compatible hash bucket. matches HashingTF(seed=42) at training."""
    if _HAVE_MMH3:
        # mmh3.hash returns a signed 32-bit int; python's % handles negatives
        return mmh3.hash(token, seed=42) % vocab_size
    # fallback: python's hash() (NOT spark-compatible, model will degrade to
    # predicting the class prior). we keep this so the dashboard still loads
    # if mmh3 is missing from the deploy env.
    return abs(hash(token)) % vocab_size

from dashboard.data_loader import load_classifier_npz, load_regressor_npz, load_json
from dashboard.charts import (
    confidence_bars, vertical_bar, empty_state,
    PURPLE_PRIMARY, GRAY_TEXT, GRAY_MUTED,
)


# regex compiled once - matches src/preprocess.py
_TOKEN_RE = re.compile(r'[^a-zA-Z]+')

# project + english stopwords. mirrors the union of nltk's english stops
# and src/preprocess.py::_PROJECT_STOPS so the deployed token preview
# matches what the trained classifier saw.
_STOPS = frozenset({
    # project-specific
    'street', 'avenue', 'ave', 'st', 'road', 'rd',
    'building', 'apt', 'apartment', 'address', 'location',
    'please', 'complaint', 'issue', 'problem',
    'nyc', 'ny', 'new', 'york',
    # high-frequency english words from nltk english stops
    'the', 'and', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'of', 'to', 'in', 'on', 'at', 'for', 'with', 'as', 'by', 'from', 'into',
    'about', 'against', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'over', 'under', 'again', 'further', 'then', 'once',
    'this', 'that', 'these', 'those', 'such', 'same', 'other',
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
    'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
    'it', 'its', 'they', 'them', 'their', 'theirs',
    'what', 'which', 'who', 'whom', 'whose',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'will', 'would', 'should', 'could', 'might', 'must', 'shall', 'may',
    'can', 'cannot', 'cant',
    'not', 'no', 'nor', 'only', 'own', 'so', 'than', 'too', 'very',
    'just', 'now', 'here', 'there', 'where', 'when', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some',
    'really', 'like', 'days', 'since', 'because', 'while', 'although',
    'one', 'two', 'three', 'first', 'last', 'still', 'yet',
})


def _tokenize(text: str) -> list:
    """tokenize like src/preprocess.py minus the lemmatization."""
    if not text:
        return []
    lower = text.lower()
    parts = [t for t in _TOKEN_RE.split(lower)
             if t and len(t) >= 3 and t not in _STOPS]
    return parts


def _predict(text: str, artifact: dict):
    """pure-python LR inference using the npz from src.classify.export_portable.
    uses mmh3 (spark-compatible hash) so token buckets match training time."""
    tokens = _tokenize(text)
    if not tokens:
        return None, [], None, []

    vocab_size = int(artifact['vocab_size'])
    raw = np.zeros(vocab_size, dtype=np.float32)
    for tok in tokens:
        # spark-compatible hashing - identical buckets to training-time
        # HashingTF(seed=42). python's built-in hash() was producing random
        # buckets per-process and degrading the classifier to predicting the
        # class prior on every input.
        idx = _bucket(tok, vocab_size)
        raw[idx] += 1.0
    weighted = raw * artifact['idf']
    logits = artifact['coefs'] @ weighted + artifact['intercepts']
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    labels = list(artifact['labels'])
    ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])
    return ranked[0][0], ranked, probs, tokens


def render() -> None:
    st.header('Triage Bot')
    st.markdown(
        '<p style="font-size: 15px; color: #4A4A4A; line-height: 1.55; max-width: 800px;">'
        'Paste an NYC 311 complaint description below. The system classifies '
        'it into one of the top 20 official categories, predicts how long it '
        'will take to resolve, and surfaces the distinctive vocabulary of the '
        'borough you select.'
        '</p>',
        unsafe_allow_html=True,
    )

    sample = (
        "There's a really bad smell coming from the trash piles on 3rd Ave, "
        "been like this for days, rats everywhere now."
    )

    col_input, col_meta = st.columns([3, 1])
    with col_input:
        text = st.text_area(
            'Complaint description',
            value=st.session_state.get('triage_text', sample),
            height=140,
            key='triage_text',
            label_visibility='visible',
        )
    with col_meta:
        borough = st.selectbox(
            'Borough',
            ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
            index=0,
            key='triage_borough',
        )
        st.write('')  # vertical spacing
        run = st.button('Triage this complaint', type='primary',
                        use_container_width=True)

    if not run:
        st.markdown(
            '<div style="margin-top: 18px; padding: 14px 18px; '
            'background: #F5F2F8; border-left: 3px solid #9B6FC2; '
            'border-radius: 4px; font-size: 14px; color: #4A4A4A;">'
            'Type or paste a complaint above and press '
            '<b style="color: #57068C;">Triage this complaint</b> to see results.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    classifier = load_classifier_npz()
    regressor = load_regressor_npz()

    # ---- classification result ----
    st.markdown('### Classification')

    if classifier is None:
        empty_state(
            'The classifier artifact is not yet published to the repo. '
            'Once <code>notebooks/FINAL_PROJECT_311_NLP.ipynb</code> runs in Colab, '
            'its auto-commit cell will push <code>models/portable/classifier.npz</code> '
            'and this section will become live.',
            action='models/portable/classifier.npz',
        )
        # still show a tokenization preview so the box isnt completely empty
        toks = _tokenize(text)
        if toks:
            st.caption('Preview - what the classifier would see after preprocessing:')
            st.markdown(
                '<div style="display: flex; flex-wrap: wrap; gap: 6px; '
                'margin-top: 8px;">' +
                ''.join(
                    f'<span style="background: #F2EFF7; color: #57068C; '
                    f'padding: 4px 10px; border-radius: 12px; font-size: 13px; '
                    f'font-family: Menlo, monospace;">{t}</span>'
                    for t in toks
                ) +
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        label, ranked, probs, toks = _predict(text, classifier)
        if label is None:
            st.warning('No usable tokens after preprocessing. Try a longer description.')
        else:
            # big result card
            st.markdown(
                f'<div style="margin-top: 8px; padding: 18px 22px; '
                f'background: linear-gradient(135deg, #57068C 0%, #3A0560 100%); '
                f'border-radius: 8px; color: white;">'
                f'<div style="font-size: 12px; font-weight: 600; '
                f'text-transform: uppercase; letter-spacing: 0.08em; '
                f'opacity: 0.85;">Predicted category</div>'
                f'<div style="font-size: 30px; font-weight: 700; '
                f'margin-top: 4px;">{label}</div>'
                f'<div style="font-size: 14px; opacity: 0.85; margin-top: 6px;">'
                f'Confidence {ranked[0][1] * 100:.1f}% &nbsp;&middot;&nbsp; '
                f'Runner-up: {ranked[1][0]} ({ranked[1][1] * 100:.1f}%)'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # top-5 confidences
            st.markdown('### Top 5 confidences')
            top5 = ranked[:5]
            fig = confidence_bars(
                [k for k, _ in top5],
                [v for _, v in top5],
                height=240,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                'Confidences are approximate - the deployed classifier uses '
                "Python's hash() for the feature space rather than the "
                "Spark-side MurmurHash3 used at training. Per-token feature "
                'indices drift slightly which softens edge-case predictions.'
            )

    # ---- resolution time ----
    st.markdown('### Predicted Resolution Time')
    if regressor is None:
        empty_state(
            'The resolution-time regressor is not yet published. Once '
            '<code>notebooks/FINAL_PROJECT_311_NLP.ipynb</code> runs (including the v2 '
            'regression block that adds <code>label_canonical</code> as a feature), '
            'its auto-commit cell will push <code>models/portable/regressor.npz</code>.',
            action='models/portable/regressor.npz',
        )
    else:
        median_map = regressor.get('median_map')
        if median_map is not None and classifier is not None and label is not None:
            try:
                lookup = dict(median_map.tolist())
                hours = float(lookup.get(label, regressor.get('global_median', 24)))
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric('Expected hours', f'{hours:.0f}')
                with col_b:
                    st.metric('Approx days', f'{hours / 24:.1f}')
                with col_c:
                    speed = 'fast' if hours < 24 else ('typical' if hours < 96 else 'slow')
                    st.metric('Speed bracket', speed)
                st.caption(
                    'Estimate is based on the median resolution time for the '
                    'predicted category in the training set.'
                )
            except Exception as e:
                st.caption(f'Could not look up median: {e}')
        else:
            st.caption('Run the classifier above first to predict resolution time.')

    # ---- borough context ----
    st.markdown('### What ' + borough + ' typically complains about')
    fingerprints = load_json('borough_fingerprints.json')
    volumes = load_json('borough_volume.json')

    if not fingerprints or borough not in fingerprints:
        empty_state(
            'Per-borough fingerprints not yet published. Run '
            '<code>notebooks/FINAL_PROJECT_311_NLP.ipynb</code> in Colab.',
            action='dashboard/assets/borough_fingerprints.json',
        )
        return

    terms = fingerprints[borough]
    fig = vertical_bar(
        labels=[t[0] for t in terms[:10]],
        values=[t[1] for t in terms[:10]],
        color=PURPLE_PRIMARY,
        height=300,
        sort_descending=True,
    )
    fig.update_layout(yaxis=dict(title=dict(text='lift', font=dict(size=11, color=GRAY_MUTED))))
    st.plotly_chart(fig, use_container_width=True)

    if volumes and borough in volumes:
        v = volumes[borough]
        st.markdown(
            f'<p style="font-size: 13px; color: {GRAY_MUTED}; margin-top: -8px;">'
            f'{borough} contributed <b style="color: {GRAY_TEXT};">{v["count"]:,}</b> '
            f'complaints to the 10M sample. '
            f'Top categories: '
            + ', '.join(f'<b style="color: {GRAY_TEXT};">{c[0]}</b>'
                        for c in v['top_categories'][:3])
            + '.'
            '</p>',
            unsafe_allow_html=True,
        )
