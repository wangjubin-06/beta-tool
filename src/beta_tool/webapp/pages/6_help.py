import streamlit as st
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.parent.parent
md_file_path = root_dir / "METHODOLOGY.md"

mdstr = Path(md_file_path).read_text(encoding="utf-8")

st.set_page_config(
    page_title="About",
    page_icon=":material/info:",
    layout="wide",
)

# Reset font family to Streamlit's default for this page only
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Source Sans Pro", sans-serif, sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    mdstr,
)