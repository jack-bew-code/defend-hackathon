"""Colours and small HTML components shared across the app."""
from html import escape

import streamlit as st

# Validated with the dataviz palette checker on the card surface (dark mode):
# lightness band, chroma, colour-blind separation and contrast all pass.
FRAUD = "#F0475E"
GENUINE = "#1FA899"
ACCENT = "#FFB703"
MUTED = "#8A9BB8"
CARD = "#13203A"
BORDER = "#22314F"
TEXT = "#E6EDF7"

CSS = f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1400px; }}
.hero h1 {{ font-size: 2.1rem; margin: 0 0 .2rem 0; }}
.hero p {{ color: {MUTED}; margin: 0 0 1rem 0; font-size: 1.05rem; }}
.card {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 1.1rem 1.25rem; height: 100%;
}}
.card .label {{ color: {MUTED}; font-size: .85rem; text-transform: uppercase; letter-spacing: .06em; }}
.card .value {{ font-size: 2.1rem; font-weight: 700; line-height: 1.2; margin-top: .25rem; }}
.card .sub {{ color: {MUTED}; font-size: .9rem; margin-top: .2rem; }}
.chip {{
  display: inline-block; padding: .28rem .7rem; margin: .18rem .3rem .18rem 0;
  border-radius: 999px; font-size: .88rem; border: 1px solid {FRAUD}55;
  background: {FRAUD}1F; color: {TEXT};
}}
.chip b {{ color: {TEXT}; font-weight: 700; }}
.rules td, .rules th {{ padding: .45rem .8rem; border-bottom: 1px solid {BORDER}; }}
.rules {{ border-collapse: collapse; width: 100%; }}
.kv {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: .6rem; }}
.kv div {{ background: #0F1A30; border: 1px solid {BORDER}; border-radius: 10px; padding: .55rem .75rem; }}
.kv span {{ display: block; color: {MUTED}; font-size: .78rem; }}
.kv strong {{ font-size: 1.05rem; }}
section[data-testid="stSidebar"] .stMarkdown p {{ color: {MUTED}; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", accent: str = BORDER) -> str:
    """Stat tile. Numbers stay in text colour; `accent` colours the left edge only."""
    sub_html = f'<div class="sub">{escape(sub)}</div>' if sub else ""
    return (f'<div class="card" style="border-left: 4px solid {accent}">'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div>{sub_html}</div>')


def cards(items: list[tuple]):
    for col, item in zip(st.columns(len(items)), items):
        col.markdown(card(*item), unsafe_allow_html=True)


def chips(pairs: list[tuple[str, str]]) -> str:
    return "".join(
        f'<span class="chip">{escape(label)}{": <b>" + escape(value) + "</b>" if value else ""}</span>'
        for label, value in pairs
    )


def key_values(pairs: list[tuple[str, str]]) -> str:
    return '<div class="kv">' + "".join(
        f"<div><span>{escape(k)}</span><strong>{escape(str(v))}</strong></div>" for k, v in pairs
    ) + "</div>"
