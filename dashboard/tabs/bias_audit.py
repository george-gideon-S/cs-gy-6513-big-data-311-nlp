"""
bias audit tab - per-borough macro-f1 and resolution-time bias.
addresses kontokosta and hong (2021) directly.
"""
import streamlit as st


def render() -> None:
    st.header("Bias Audit")
    st.write(
        "Models trained on 311 inherit reporting biases from the underlying "
        "complaint behavior. Following Kontokosta & Hong (2021, *Sustainable "
        "Cities and Society*), this tab surfaces where our classifier and "
        "resolution-time model perform unevenly across boroughs."
    )
    st.info(
        "Bias slices are computed at the end of Phase 3 / Phase 4 and saved "
        "to `dashboard/assets/bias_metrics.json`. Once that file exists, "
        "this tab will render per-borough F1 and predicted-vs-actual scatter."
    )
