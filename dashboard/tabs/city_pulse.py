"""
city pulse - choropleth of complaint volume per borough + per-borough
top categories. drives off the phase 6 artifacts.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import load_geojson, load_json, render_missing


def render() -> None:
    st.header('City Pulse')
    st.write(
        'Where complaints concentrate across the five boroughs and what '
        "categories dominate each one. Aggregated from the 2M-row sample "
        'of NYC 311 complaints (2010-present).'
    )

    geo = load_geojson()
    volumes = load_json('borough_volume.json')

    if not geo or not volumes:
        render_missing('Phase 6', '06_geo_census.ipynb',
                       what='The City Pulse map')
        return

    # 1. choropleth of complaint volume
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.error('streamlit-folium not installed. Add it to requirements.txt and redeploy.')
        return

    # build a folium choropleth
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles='cartodbpositron')

    # the geojson features have property `name` matching the borough names in volumes
    folium.Choropleth(
        geo_data=geo,
        data={k: v['count'] for k, v in volumes.items()},
        key_on='feature.properties.name',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name='Complaint volume (sample)',
        nan_fill_color='lightgray',
    ).add_to(m)

    # add tooltip layer that shows count on hover
    def style_fn(_feature):
        return {'fillOpacity': 0, 'color': 'transparent', 'weight': 0}

    def hl_fn(_feature):
        return {'fillOpacity': 0.4, 'fillColor': '#666', 'color': '#222', 'weight': 2}

    folium.GeoJson(
        geo,
        style_function=style_fn,
        highlight_function=hl_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=['name'],
            aliases=['Borough'],
            localize=True,
        ),
    ).add_to(m)

    st.subheader('Complaint volume by borough')
    st_folium(m, height=480, use_container_width=True, returned_objects=[])

    # 2. selector + per-borough top categories
    st.subheader('Top complaint categories per borough')
    boroughs = sorted(volumes.keys(), key=lambda b: -volumes[b]['count'])
    cols = st.columns(len(boroughs))
    for col, b in zip(cols, boroughs):
        with col:
            v = volumes[b]
            st.metric(b, f"{v['count']:,}")
            for name, cnt in v['top_categories'][:5]:
                st.caption(f'**{name}** ({cnt:,})')
