"""
Alternate visual theme for the beta_tool webapp — "terminal desk" instead of
"paper ledger". Same structure and public API as theme.py (same function
names/signatures), so this is a drop-in swap for side-by-side comparison:

    # to try it:
    from beta_tool.webapp import theme_terminal as theme
    theme.apply_theme()

    # or just rename this file to theme.py (backing up the original first)

Where theme.py leans on a warm cream page + serif headings (editorial /
research-note feel), this one leans on a cool near-black page + monospace
headings (trading-terminal / HUD feel) — amber as the primary accent (the
classic terminal color), teal as the secondary line color, with green/red
reserved strictly for positive/negative so they read as signal, not decor.

Native Streamlit theming (colors, fonts, radii, widget chrome) still lives
in .streamlit/config.toml under [theme] / [theme.light] / [theme.dark] —
this module covers what config.toml can't reach. See the companion
theme_terminal_config.toml for a starting-point [theme] block matching the
PALETTE below; merge it into your existing config rather than overwriting it,
since native widget chrome (buttons, sliders, inputs) won't match this
module's CSS/Plotly colors until config.toml is updated too.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

FONT_BODY = "Inter, sans-serif"
FONT_HEADING = "Space Grotesk, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"

PALETTE = {
    "light": {
        "bg": "#F4F6F8",
        "surface": "#FFFFFF",
        "ink": "#12161C",
        "muted": "#5B6472",
        "border": "#D7DCE3",
        "navy": "#C77D0A",
        "clay": "#0F766E",
        "positive": "#15803D",
        "negative": "#B91C1C",
        "grid": "#E4E8ED",
        "shadow": "rgba(15, 23, 32, 0.12)",
    },
    "dark": {
        "bg": "#0B0D10",
        "surface": "#15181D",
        "ink": "#E8EAED",
        "muted": "#8A93A3",
        "border": "#262B33",
        "navy": "#FFB020",
        "clay": "#2DD4BF",
        "positive": "#22C55E",
        "negative": "#EF4444",
        "grid": "#1C2027",
        "shadow": "rgba(0, 0, 0, 0.55)",
    },
}

TEMPLATE_NAME = "quant_terminal"


def get_theme_mode() -> str:
    """Best-effort read of the viewer's active theme, 'light' or 'dark'.

    st.context.theme.type shipped in Streamlit 1.46 and can briefly read as
    None on an app's very first run in a session (known upstream quirk), so
    this always falls back to 'dark' — a terminal theme should default to
    dark, not light, if the viewer's preference can't be read yet.
    """
    try:
        mode = st.context.theme.type
    except Exception:
        mode = None
    return mode if mode in PALETTE else "dark"


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
    """Register the 'quant_terminal' Plotly template for the current theme
    mode and set it as Plotly's *default* template — this is what makes
    px/go charts pick up the app's colors without every call needing a
    `template=` argument. Also returns the name, if you'd rather be
    explicit: `fig.update_layout(template=theme.quant_template())`.

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
    font-family: {mono};
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    opacity: 0.85;
}}
[data-testid="stSidebarNav"] {{
    font-family: {mono};
}}
hr {{
    margin: 1.25rem 0;
    opacity: 0.4;
}}

[data-testid="baseButton-primary"],
[data-testid="stBaseButton-primary"] {{
    font-family: {mono};
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
            shadow=p["shadow"],
        ),
        unsafe_allow_html=True,
    )
    quant_template()