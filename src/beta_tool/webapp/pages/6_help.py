import streamlit as st
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.parent.parent
md_file_path = root_dir / "METHODOLOGY.md"

mdstr = Path(md_file_path).read_text(encoding="utf-8")
modstr = f'<span style="font-family: sans-serif;">{mdstr}</span>'

st.set_page_config(
    page_title="About",
    page_icon=":material/info:",
    layout="wide",
)

st.markdown(
    modstr,
    unsafe_allow_html=True,
)