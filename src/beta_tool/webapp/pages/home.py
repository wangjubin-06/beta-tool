import streamlit as st


st.set_page_config(
    page_title="beta-tool",
    page_icon="🚀", # Can be an emoji or a path to an image file
    layout="centered"
)

st.title("beta_tool")
st.write(
    "A quantitative finance toolkit for beta estimation, portfolio hedging, "
    "and factor analysis. Select a tool from the sidebar."
)

st.subheader("Available tools")
st.markdown("""
- **Beta** — two-asset return regression
- **MultiBeta** — multi-asset regression
- **Portfolio Beta** — weighted portfolio vs benchmark asset(s)
- **Hedge** — beta-weighted hedge construction and backtesting
- **Factors** — equity vs factor regression
""")