"""
chart helpers used across all tabs. all charts use the project palette
(NYU purple at varying weights) and a consistent typography stack.

we render with plotly express rather than st.bar_chart because:
1. st.bar_chart sorts categorical x values alphabetically, breaking
   any plot where order matters (e.g., "top terms by lift").
2. it doesn't expose enough styling control for a professional look.
3. plotly handles long category labels with rotation/truncation cleanly.
"""
from __future__ import annotations
from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# project palette - NYU purple primary plus a sequential ramp
PURPLE_PRIMARY = '#57068C'
PURPLE_LIGHT = '#9B6FC2'
PURPLE_DEEP = '#3A0560'
ACCENT_ORANGE = '#FF6F00'
GRAY_TEXT = '#4A4A4A'
GRAY_MUTED = '#8A8A8A'

# sequential color ramp for choropleths and gradient bars
PURPLE_RAMP = [
    '#F2EFF7', '#E6DDF1', '#C9B0E1', '#9B6FC2',
    '#7A33B2', '#57068C', '#3A0560',
]


def _base_layout(height: int = 320, show_xgrid: bool = False) -> dict:
    """consistent plotly layout shared across charts."""
    return dict(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, -apple-system, sans-serif',
                  size=13, color=GRAY_TEXT),
        xaxis=dict(
            showgrid=show_xgrid,
            gridcolor='#F0F0F0',
            showline=True,
            linecolor='#E0E0E0',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#F0F0F0',
            zeroline=False,
            showline=False,
        ),
        showlegend=False,
    )


def horizontal_bar(
    labels: Sequence[str],
    values: Sequence[float],
    title: str | None = None,
    height: int = 360,
    color: str = PURPLE_PRIMARY,
    value_format: str = '.2f',
    x_label: str | None = None,
    y_label: str | None = None,
) -> go.Figure:
    """
    horizontal bars sorted descending by value. each bar shows its value
    at the right edge so the user doesnt have to read the axis.
    """
    df = pd.DataFrame({'label': list(labels), 'value': list(values)})
    df = df.sort_values('value', ascending=True)  # ascending here = top of chart shows largest

    fig = go.Figure(go.Bar(
        x=df['value'],
        y=df['label'],
        orientation='h',
        marker=dict(color=color, line=dict(width=0)),
        text=[f'{v:{value_format}}' for v in df['value']],
        textposition='outside',
        textfont=dict(size=12, color=GRAY_TEXT),
        hovertemplate='<b>%{y}</b>: %{x:.2f}<extra></extra>',
    ))
    layout = _base_layout(height=height, show_xgrid=True)
    layout['yaxis'] = dict(
        showgrid=False,
        showline=False,
        ticks='',
        automargin=True,
    )
    layout['xaxis'] = dict(
        showgrid=True,
        gridcolor='#F0F0F0',
        showline=False,
        zeroline=False,
        title=dict(text=x_label or '', font=dict(size=12, color=GRAY_MUTED)),
    )
    if title:
        layout['title'] = dict(text=title, font=dict(size=16, color=PURPLE_DEEP),
                               x=0, xanchor='left')
    fig.update_layout(**layout)
    return fig


def vertical_bar(
    labels: Sequence[str],
    values: Sequence[float],
    title: str | None = None,
    height: int = 320,
    color: str = PURPLE_PRIMARY,
    value_format: str = '.2f',
    sort_descending: bool = True,
) -> go.Figure:
    """
    vertical bars in user-supplied order (or sorted). use this when
    rotated x labels look fine, e.g., for ~10 items.
    """
    df = pd.DataFrame({'label': list(labels), 'value': list(values)})
    if sort_descending:
        df = df.sort_values('value', ascending=False)

    fig = go.Figure(go.Bar(
        x=df['label'],
        y=df['value'],
        marker=dict(color=color, line=dict(width=0)),
        text=[f'{v:{value_format}}' for v in df['value']],
        textposition='outside',
        textfont=dict(size=11, color=GRAY_TEXT),
        hovertemplate='<b>%{x}</b>: %{y:.2f}<extra></extra>',
    ))
    layout = _base_layout(height=height, show_xgrid=False)
    layout['xaxis']['categoryorder'] = 'array'
    layout['xaxis']['categoryarray'] = df['label'].tolist()
    layout['xaxis']['tickangle'] = -35
    if title:
        layout['title'] = dict(text=title, font=dict(size=16, color=PURPLE_DEEP),
                               x=0, xanchor='left')
    fig.update_layout(**layout)
    return fig


def confidence_bars(
    labels: Sequence[str],
    probs: Sequence[float],
    height: int = 320,
) -> go.Figure:
    """
    horizontal bars showing classifier confidences. the top class gets
    full purple; others fade to a lighter shade so the prediction
    visually pops out.
    """
    df = pd.DataFrame({'label': list(labels), 'prob': list(probs)})
    df = df.sort_values('prob', ascending=True)
    # darkest for the top, fading down
    n = len(df)
    colors = [PURPLE_PRIMARY if i == n - 1 else PURPLE_LIGHT for i in range(n)]

    fig = go.Figure(go.Bar(
        x=df['prob'],
        y=df['label'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{p * 100:.1f}%' for p in df['prob']],
        textposition='outside',
        textfont=dict(size=12, color=GRAY_TEXT),
        hovertemplate='<b>%{y}</b>: %{x:.3f}<extra></extra>',
    ))
    layout = _base_layout(height=height, show_xgrid=True)
    layout['yaxis'] = dict(showgrid=False, showline=False, ticks='', automargin=True)
    layout['xaxis']['range'] = [0, max(df['prob'].max() * 1.15, 0.1)]
    layout['xaxis']['title'] = dict(text='confidence', font=dict(size=11, color=GRAY_MUTED))
    fig.update_layout(**layout)
    return fig


def empty_state(message: str, action: str | None = None) -> None:
    """
    consistent placeholder used wherever an artifact is missing. no
    yellow warning flash - just a small clean note.
    """
    border = f'border-left: 3px solid {PURPLE_LIGHT}'
    block = (
        f'<div style="{border}; padding: 14px 18px; '
        f'background: #FAFAFC; margin: 12px 0; border-radius: 4px;">'
        f'<div style="color: {GRAY_TEXT}; font-size: 14px; line-height: 1.5;">'
        f'{message}'
        f'</div>'
    )
    if action:
        block += (
            f'<div style="color: {GRAY_MUTED}; font-size: 12px; '
            f'margin-top: 6px; font-family: Menlo, monospace;">'
            f'{action}'
            f'</div>'
        )
    block += '</div>'
    st.markdown(block, unsafe_allow_html=True)
