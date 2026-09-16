"""
Shared visual theme for the beta_tool webapp.

Native Streamlit theming (colors, fonts, radii, widget chrome) lives in
.streamlit/config.toml under [theme] / [theme.light] / [theme.dark]. This
module covers what config.toml can't reach:

1. CSS touch-ups (tabular numerals in st.metric, heading tracking, sidebar
   label styling, card-style metrics with a hover lift) via `apply_theme()`.
2. A Plotly template that tracks the viewer's active mode and is set as
   Plotly's *default*, so every chart — px or go, whether or not it passes
   a `template=` argument — follows the app's colors automatically.

Usage
-----
    # app.py, before any other st.* calls that render content:
    from beta_tool.webapp import theme
    theme.apply_theme()

That's it — apply_theme() also syncs the Plotly default template. Drop any
leftover `template="plotly_white"` (or similar) arguments in page code; an
explicit argument always overrides the default.

Keep PALETTE below in sync with .streamlit/config.toml if you retune colors
— Streamlit doesn't expose a supported way to read resolved theme colors
back out in Python, so this is a second source of truth by necessity, not
an oversight.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

FONT_BODY = "IBM Plex Sans, sans-serif"
FONT_HEADING = "Source Serif 4, serif"
FONT_MONO = "IBM Plex Mono, monospace"

PALETTE = {
    "light": {
        "bg": "#F7F5F0",
        "surface": "#EDEAE1",
        "ink": "#22201C",
        "muted": "#6B6659",
        "border": "#DDD9CE",
        "navy": "#2B4570",
        "clay": "#D9663A",
        "positive": "#1F6F50",
        "negative": "#A23B2E",
        "grid": "#E3DFD4",
        "shadow": "rgba(34, 32, 28, 0.10)",
    },
    "dark": {
        "bg": "#131211",
        "surface": "#1D1B18",
        "ink": "#F5F2EA",
        "muted": "#9A9284",
        "border": "#3D372C",
        "navy": "#8FB4E3",
        "clay": "#F2794D",
        "positive": "#35D68E",
        "negative": "#FF6B5B",
        "grid": "#241F19",
        "shadow": "rgba(0, 0, 0, 0.45)",
    },
}

TEMPLATE_NAME = "quant"


def get_theme_mode() -> str:
    """Best-effort read of the viewer's active theme, 'light' or 'dark'.

    st.context.theme.type shipped in Streamlit 1.46 and can briefly read as
    None on an app's very first run in a session (known upstream quirk), so
    this always falls back to 'light' instead of raising.
    """
    try:
        mode = st.context.theme.type
    except Exception:
        mode = None
    return mode if mode in PALETTE else "light"


def signed_color(value: float, mode: str | None = None) -> str:
    """Hex color for a signed number (beta delta, P&L, etc.), theme-matched.

    Positive -> green, negative -> red, zero -> muted ink. Handy for
    coloring custom badges/captions outside of st.metric's built-in delta.
    """
    p = PALETTE[mode or get_theme_mode()]
    if value > 0:
        return p["positive"]
    if value < 0:
        return p["negative"]
    return p["muted"]


def build_plotly_template(mode: str) -> go.layout.Template:
    p = PALETTE[mode]
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_BODY, color=p["ink"], size=13),
            title=dict(font=dict(family=FONT_HEADING, size=20, color=p["ink"])),
            paper_bgcolor=p["bg"],
            plot_bgcolor=p["bg"],
            colorway=[p["navy"], p["clay"], p["positive"], p["negative"], p["muted"]],
            xaxis=dict(
                gridcolor=p["grid"],
                zerolinecolor=p["border"],
                linecolor=p["border"],
                tickfont=dict(family=FONT_MONO, size=11, color=p["muted"]),
            ),
            yaxis=dict(
                gridcolor=p["grid"],
                zerolinecolor=p["border"],
                linecolor=p["border"],
                tickfont=dict(family=FONT_MONO, size=11, color=p["muted"]),
            ),
            legend=dict(font=dict(family=FONT_BODY, size=12, color=p["ink"])),
            hoverlabel=dict(
                font=dict(family=FONT_MONO, size=12),
                bgcolor=p["surface"],
                bordercolor=p["border"],
            ),
            margin=dict(t=60, l=10, r=10, b=10),
        )
    )


def quant_template() -> str:
    """Register the 'quant' Plotly template for the current theme mode and
    set it as Plotly's *default* template — this is what makes px/go charts
    pick up the app's colors without every call needing a `template=`
    argument. Also returns the name, if you'd rather be explicit:
    `fig.update_layout(template=theme.quant_template())`.

    Called automatically by `apply_theme()`; exposed separately in case a
    page wants to re-sync mid-script (e.g. inside a fragment).
    """
    pio.templates[TEMPLATE_NAME] = build_plotly_template(get_theme_mode())
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


_CSS = """
<style>
[data-testid="stMetricValue"] {{
    font-family: {mono};
    font-variant-numeric: tabular-nums;
}}
h1, h2, h3 {{
    font-family: {heading};
    letter-spacing: -0.01em;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {{
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.85;
}}
[data-testid="stSidebarNav"] {{
    font-family: {body};
}}
hr {{
    margin: 1.25rem 0;
}}

/* card-style metrics with a soft hover lift — the "modern SaaS" bit */

/* [data-testid="stMetric"] {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
}} */

[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px {shadow};
    border-color: {primary};
}}
[data-testid="baseButton-primary"],
[data-testid="stBaseButton-primary"] {{
    transition: transform 120ms ease, box-shadow 120ms ease;
}}
[data-testid="baseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 16px {shadow};
}}
</style>
"""


def apply_theme() -> None:
    """Inject the app-wide CSS touch-ups and sync Plotly's default template
    to the active mode. Call once, at the top of app.py, before any other
    st.* calls that render content.
    """
    p = PALETTE[get_theme_mode()]
    st.markdown(
        _CSS.format(
            mono=FONT_MONO,
            heading=FONT_HEADING,
            body=FONT_BODY,
            surface=p["surface"],
            border=p["border"],
            primary=p["clay"],
            shadow=p["shadow"],
        ),
        unsafe_allow_html=True,
    )
    quant_template()