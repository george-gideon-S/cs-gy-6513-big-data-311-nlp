"""
city pulse tab - choropleth + time series of complaint volume.
shown by district when phase 6 is done; falls back to borough until then.
"""
import streamlit as st


def render() -> None:
    st.header("City Pulse")
    st.write(
        "Where are complaints happening, and when. The choropleth highlights "
        "districts with above-average complaint density; the time series "
        "tracks daily/weekly volume."
    )
    st.info(
        "Geographic + temporal aggregation lands in Phase 6 + Phase 9. "
        "Once `notebooks/06_geo_census.ipynb` writes "
        "`delta/district_volume.parquet`, this tab will render the map."
    )
