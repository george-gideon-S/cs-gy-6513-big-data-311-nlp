"""
live pulse - tails the structured-streaming sink + offers a 'fetch fresh from
soda api' button so we can demo near-real-time classification without needing
spark in the deployed environment.
"""
import streamlit as st


def render() -> None:
    st.header("Live Pulse")
    st.write(
        "Spark Structured Streaming proof-of-concept. Drop a CSV into the "
        "watch folder on Colab, and within one micro-batch its rows are "
        "classified and appear here."
    )
    st.button("Refresh from SODA API", help="Pull the last 100 complaints from NYC Open Data and classify them live.")
    st.info(
        "Streaming sink and SODA refresh land in Phase 8. Once "
        "`delta/streaming_out/` exists, this tab will show a live-updating "
        "table of recent classifications."
    )
