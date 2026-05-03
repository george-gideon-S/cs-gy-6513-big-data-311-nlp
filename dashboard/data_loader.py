"""
artifact loaders for the streamlit app. all use st.cache_data /
st.cache_resource so they only run once per session.

resolution order for json artifacts:
  1. on-disk file at dashboard/assets/<name>.json (the freshest data,
     pushed by a recent notebook run)
  2. dashboard.baked_in_data.BAKED_IN[<name>] (hardcoded fallback with
     real numbers from the most recent successful end-to-end run; means
     the dashboard always renders even when no JSON has been pushed)
  3. None (only happens for artifacts we have no baked-in fallback for,
     such as the geojson)

binary models (.npz, .kv) cannot be inlined cleanly so they still
return None when the file is missing. tabs that need them (Triage Bot
live inference, Cluster Atlas synonym probes) handle that case.
"""
from __future__ import annotations
import json
from pathlib import Path

import streamlit as st

from dashboard.baked_in_data import BAKED_IN


_BASE = Path(__file__).resolve().parents[1]
_ASSETS = _BASE / 'dashboard' / 'assets'
_PORTABLE = _BASE / 'models' / 'portable'


def asset_path(name: str) -> Path:
    return _ASSETS / name


def portable_path(name: str) -> Path:
    return _PORTABLE / name


@st.cache_data
def load_json(name: str):
    """
    load a JSON artifact from disk, or fall back to the hardcoded
    equivalent in dashboard.baked_in_data. returns None only if neither
    is available (e.g. nyc_boroughs.geojson which isnt baked in - we
    keep that one as a real file in the repo).
    """
    p = _ASSETS / name
    # 1) prefer the real file if its there
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            pass  # fall through to baked-in
    # 2) baked-in fallback. strip .json suffix to look up the key.
    key = name[:-5] if name.endswith('.json') else name
    return BAKED_IN.get(key)


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


def _json_available(name: str) -> bool:
    """a JSON artifact counts as available if its on disk OR baked in."""
    if asset_path(name).exists():
        return True
    key = name[:-5] if name.endswith('.json') else name
    return key in BAKED_IN


def all_artifacts_present() -> dict:
    """quick check used by the pipeline-status tab."""
    return {
        # phase 6
        'phase_6_geo_summary': _json_available('geo_summary.json'),
        'phase_6_borough_volume': _json_available('borough_volume.json'),
        'phase_6_borough_fingerprints': _json_available('borough_fingerprints.json'),
        'phase_6_boroughs_geojson': asset_path('nyc_boroughs.geojson').exists(),
        # phase 3
        'phase_3_classifier_summary': _json_available('classifier_summary.json'),
        'phase_3_classifier_portable': portable_path('classifier.npz').exists(),
        'phase_3_per_class_metrics': _json_available('per_class_metrics.json'),
        'phase_3_confusion_matrix': asset_path('cm.png').exists(),
        # phase 4
        'phase_4_regressor_summary': _json_available('regressor_summary.json'),
        'phase_4_regressor_portable': portable_path('regressor.npz').exists(),
        'phase_4_per_category_mae': _json_available('regress_by_category.json'),
        # phase 5
        'phase_5_cluster_summary': _json_available('cluster_summary.json'),
        'phase_5_word2vec_kv': portable_path('word2vec.kv').exists(),
        'phase_5_silhouette_curve': asset_path('silhouette_curve.png').exists(),
        # phase 10
        'phase_10_bert_summary': _json_available('bert_cluster_summary.json'),
        'phase_10_bert_vs_w2v_plot': asset_path('bert_vs_w2v.png').exists(),
        # phase 1/2 dataset distribution
        'phase_1_class_dist_png': asset_path('class_dist.png').exists(),
        'phase_2_preprocess_stats': _json_available('preprocess_stats.json'),
    }


def render_missing(phase: str, notebook: str, what: str = 'this tab') -> None:
    """consistent placeholder shown when a tab's artifacts are not yet pushed."""
    st.warning(
        f"{what} is waiting on {phase}. To enable, open "
        f"`notebooks/{notebook}` in Colab, run all cells, and the "
        f"auto-commit cell at the end will push the required artifacts."
    )
