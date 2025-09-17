"""Theme utilities for light/dark responsive design."""

from __future__ import annotations

from dataclasses import dataclass

import plotly.io as pio
import streamlit as st


@dataclass(frozen=True)
class ThemePalette:
    """Container for theme colours compliant with WCAG 2.1 AA."""

    background: str
    surface: str
    text: str
    muted: str
    accent: str
    accent_soft: str


LIGHT_THEME = ThemePalette(
    background="#f7f9fb",
    surface="#ffffff",
    text="#1c2733",
    muted="#4f5d75",
    accent="#0b5394",
    accent_soft="#3d85c6",
)

DARK_THEME = ThemePalette(
    background="#0c1724",
    surface="#14263c",
    text="#f5f7fa",
    muted="#a0aec0",
    accent="#6cb1ff",
    accent_soft="#89c2ff",
)


COLORWAY = [
    "#0b5394",
    "#3d85c6",
    "#073763",
    "#6fa8dc",
    "#2c7fb8",
    "#012a4a",
]


def apply_streamlit_theme(mode: str) -> None:
    """Inject CSS variables into the Streamlit app for responsiveness."""

    palette = LIGHT_THEME if mode == "light" else DARK_THEME
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {palette.background};
            --surface: {palette.surface};
            --text: {palette.text};
            --muted: {palette.muted};
            --accent: {palette.accent};
            --accent-soft: {palette.accent_soft};
        }}
        body, .stApp, [data-testid="stAppViewContainer"] {{
            background-color: var(--bg) !important;
            color: var(--text) !important;
        }}
        .stMetric, .stContainer, .stDataFrame {{
            border-radius: 16px;
            background-color: var(--surface);
            color: var(--text);
        }}
        .chart-card {{
            background-color: var(--surface);
            border-radius: 18px;
            border: 1px solid rgba(0, 0, 0, 0.04);
            padding: 1rem;
        }}
        .section-title {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
            color: var(--accent);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    template = {
        "layout": {
            "font": {"color": palette.text},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "colorway": COLORWAY,
            "legend": {"orientation": "h", "y": -0.2},
        }
    }
    pio.templates["rolling_theme"] = template
    pio.templates.default = "plotly+rolling_theme"


def sidebar_mode_toggle() -> str:
    """Render the light/dark theme switcher in the sidebar."""

    default = st.session_state.get("theme_mode", "light")
    mode = st.sidebar.radio(
        "表示モード",
        options=["light", "dark"],
        format_func=lambda v: "ライト" if v == "light" else "ダーク",
        index=["light", "dark"].index(default if default in ("light", "dark") else "light"),
        help="WCAG 2.1対応の配色を選択します。",
    )
    st.session_state["theme_mode"] = mode
    return mode
