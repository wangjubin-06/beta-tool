import streamlit as st
from pathlib import Path

mdstr = Path("/METHODOLOGY.md").read_text(encoding="utf-8")

st.set_page_config(
    page_title="About",
    page_icon=":material/info:",
    layout="wide",
)


st.title("Explanation of this tool")
st.space("small")


st.markdown(
    mdstr,
)