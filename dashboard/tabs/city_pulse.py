"""
city pulse - choropleth of complaint volume per borough + per-borough
top categories.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import load_geojson, load_json
from dashboard.charts import empty_state, PURPLE_PRIMARY, PURPLE_RAMP, GRAY_MUTED, GRAY_TEXT


def render() -> None:
    st.header('City Pulse')
    st.markdown(
        '<p style="font-size: 15px; color: #4A4A4A; line-height: 1.55; max-width: 800px;">'
        'Where complaints concentrate across the five boroughs and what '
        'categories dominate each one. Aggregated from the 2 million row '
        'stratified sample of NYC 311 complaints.'
        '</p>',
        unsafe_allow_html=True,
    )

    geo = load_geojson()
    volumes = load_json('borough_volume.json')

    if not geo or not volumes:
        empty_state(
            'Borough volume data not yet published. Run '
            '<code>notebooks/06_geo_census.ipynb</code> in Colab.',
            action='dashboard/assets/borough_volume.json',
        )
        return

    # ---- top-line metrics ----
    total = sum(v['count'] for v in volumes.values())
    top_b = max(volumes.items(), key=lambda kv: kv[1]['count'])
    bot_b = min(volumes.items(), key=lambda kv: kv[1]['count'])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric('Complaints in sample', f'{total:,}')
    with m2:
        st.metric('Highest-volume borough',
                  top_b[0],
                  delta=f'{top_b[1]["count"]:,} complaints',
                  delta_color='off')
    with m3:
        st.metric('Lowest-volume borough',
                  bot_b[0],
                  delta=f'{bot_b[1]["count"]:,} complaints',
                  delta_color='off')

    # ---- choropleth ----
    st.markdown('### Complaint volume by borough')
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.error('streamlit-folium missing from requirements.txt')
        return

    m = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=10,
        tiles='cartodbpositron',
    )

    folium.Choropleth(
        geo_data=geo,
        data={k: v['count'] for k, v in volumes.items()},
        key_on='feature.properties.name',
        fill_color='Purples',
        fill_opacity=0.78,
        line_opacity=0.4,
        line_color='#FFFFFF',
        legend_name='Complaint volume',
        nan_fill_color='lightgray',
    ).add_to(m)

    # transparent overlay just for hover tooltips
    folium.GeoJson(
        geo,
        style_function=lambda f: {'fillOpacity': 0, 'color': 'transparent', 'weight': 0},
        highlight_function=lambda f: {'fillOpacity': 0.25, 'fillColor': '#57068C',
                                       'color': '#3A0560', 'weight': 2},
        tooltip=folium.GeoJsonTooltip(
            fields=['name'], aliases=['Borough'], localize=True,
            sticky=False,
            style='background: white; color: #57068C; font-weight: 600; '
                  'padding: 4px 8px; border-radius: 4px; font-size: 13px;',
        ),
    ).add_to(m)

    st_folium(m, height=460, use_container_width=True, returned_objects=[])

    # ---- per-borough top categories ----
    st.markdown('### Top complaint categories per borough')
    boroughs = sorted(volumes.keys(), key=lambda b: -volumes[b]['count'])
    cols = st.columns(len(boroughs))
    for col, b in zip(cols, boroughs):
        with col:
            v = volumes[b]
            # custom card markup so all 5 visually balance
            cats_html = ''
            for name, cnt in v['top_categories'][:5]:
                cats_html += (
                    f'<div style="display: flex; justify-content: space-between; '
                    f'padding: 6px 0; border-bottom: 1px solid #F0F0F0; '
                    f'font-size: 12px;">'
                    f'<span style="color: {GRAY_TEXT}; font-weight: 500; '
                    f'overflow: hidden; text-overflow: ellipsis; white-space: nowrap; '
                    f'margin-right: 8px;" title="{name}">{name}</span>'
                    f'<span style="color: {GRAY_MUTED}; font-variant-numeric: tabular-nums;">'
                    f'{cnt:,}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background: #FAFAFC; padding: 14px 16px; '
                f'border-radius: 6px; border: 1px solid #EDE8F2; height: 100%;">'
                f'<div style="font-size: 12px; color: {GRAY_MUTED}; '
                f'font-weight: 600; text-transform: uppercase; '
                f'letter-spacing: 0.06em;">{b}</div>'
                f'<div style="font-size: 24px; color: {PURPLE_PRIMARY}; '
                f'font-weight: 700; margin: 4px 0 8px 0;">{v["count"]:,}</div>'
                f'{cats_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
