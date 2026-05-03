"""
artifact loaders for the streamlit app. all use st.cache_data /
st.cache_resource so they only run once per session.

each function returns None if the artifact is missing, so tabs can
gracefully degrade to a placeholder. as the user re-runs phase notebooks
and the auto-commit cells push artifacts to the repo, the cache misses
on the next streamlit cloud rebuild and the tabs light up.
"""
from __future__ import annotations
import json
from pathlib import Path

import streamlit as st


_BASE = Path(__file__).resolve().parents[1]
_ASSETS = _BASE / 'dashboard' / 'assets'
_PORTABLE = _BASE / 'models' / 'portable'


def asset_path(name: str) -> Path:
    return _ASSETS / name


def portable_path(name: str) -> Path:
    return _PORTABLE / name


@st.cache_data
def load_json(name: str) -> dict | None:
    p = _ASSETS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


@st.cache_data
def load_geojson() -> dict | None:
    return load_json('nyc_boroughs.geojson')


@st.cache_resource
def load_classifier_npz():
    """returns the loaded npz dict, or None if the artifact is missing."""
    p = _PORTABLE / 'classifier.npz'
    if not p.exists():
        return None
    import numpy as np
    return dict(np.load(p, allow_pickle=True))


@st.cache_resource
def load_regressor_npz():
    p = _PORTABLE / 'regressor.npz'
    if not p.exists():
        return None
    import numpy as np
    return dict(np.load(p, allow_pickle=True))


@st.cache_resource
def load_word2vec_kv():
    """gensim is optional at deploy time. if its not installed (default
    for streamlit cloud), this returns None and tabs degrade gracefully."""
    p = _PORTABLE / 'word2vec.kv'
    if not p.exists():
        return None
    try:
        import gensim
        return gensim.models.KeyedVectors.load(str(p))
    except ImportError:
        # gensim not in deploy requirements - thats expected
        return None
    except Exception:
        return None


def all_artifacts_present() -> dict:
    """quick check used by the pipeline-status tab."""
    return {
        # phase 6
        'phase_6_geo_summary': asset_path('geo_summary.json').exists(),
        'phase_6_borough_volume': asset_path('borough_volume.json').exists(),
        'phase_6_borough_fingerprints': asset_path('borough_fingerprints.json').exists(),
        'phase_6_boroughs_geojson': asset_path('nyc_boroughs.geojson').exists(),
        # phase 3
        'phase_3_classifier_summary': asset_path('classifier_summary.json').exists(),
        'phase_3_classifier_portable': portable_path('classifier.npz').exists(),
        'phase_3_per_class_metrics': asset_path('per_class_metrics.json').exists(),
        'phase_3_confusion_matrix': asset_path('cm.png').exists(),
        # phase 4
        'phase_4_regressor_summary': asset_path('regressor_summary.json').exists(),
        'phase_4_regressor_portable': portable_path('regressor.npz').exists(),
        'phase_4_per_category_mae': asset_path('regress_by_category.json').exists(),
        # phase 5
        'phase_5_cluster_summary': asset_path('cluster_summary.json').exists(),
        'phase_5_word2vec_kv': portable_path('word2vec.kv').exists(),
        'phase_5_silhouette_curve': asset_path('silhouette_curve.png').exists(),
        # phase 10
        'phase_10_bert_summary': asset_path('bert_cluster_summary.json').exists(),
        'phase_10_bert_vs_w2v_plot': asset_path('bert_vs_w2v.png').exists(),
        # phase 1/2 dataset distribution
        'phase_1_class_dist_png': asset_path('class_dist.png').exists(),
        'phase_2_preprocess_stats': asset_path('preprocess_stats.json').exists(),
    }


def render_missing(phase: str, notebook: str, what: str = 'this tab') -> None:
    """consistent placeholder shown when a tab's artifacts are not yet pushed."""
    st.warning(
        f"{what} is waiting on {phase}. To enable, open "
        f"`notebooks/{notebook}` in Colab, run all cells, and the "
        f"auto-commit cell at the end will push the required artifacts."
    )
