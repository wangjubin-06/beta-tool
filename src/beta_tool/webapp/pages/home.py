import streamlit as st


st.set_page_config(
    page_title="Home",
    page_icon="🚀", # Can be an emoji or a path to an image file
    layout="wide"
)

st.title("beta_tool")
st.write(
    "A quantitative finance toolkit for beta estimation, portfolio hedging, "
    "and factor analysis. Select a tool from the sidebar."
)

st.subheader("Available tools")
st.markdown("""
- **Single Asset Beta** — two-asset return regression
- **Beta Tool** — multi-asset regression
- **Portfolio Beta** — weighted portfolio vs benchmark asset(s)
- **Hedge** — beta-weighted hedge construction and backtesting
- **Factor Analysis** — equity vs factor regression
""")

st.markdown("")

with st.container(width='content', gap="xxsmall"):
    st.markdown("Made by Jubin Wang")
    st.markdown("Socials:")

with st.container(width='content', gap='xsmall', horizontal=True):
    st.link_button("GitHub", "https://github.com/wangjubin-06", type='primary', width='stretch')
    st.link_button("LinkedIn", "https://linkedin.com/in/jubin-wang", type='primary', width='stretch')
    st.link_button("Instagram", "https://instagram.com/jubin.w", type='primary', width='stretch')