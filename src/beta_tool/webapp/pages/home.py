import streamlit as st
from beta_tool.webapp.tickers import is_valid_tiingo_key


st.set_page_config(
    page_title="Home",
    page_icon=":material/query_stats:",
    layout="centered"
)


with st.container(height='content'):
    st.title("beta_tool")
    st.markdown(
        "##### A quantitative finance toolkit for beta estimation, portfolio hedging, "
        "and factor analysis. Select a tool from the sidebar."
    )
    st.space("xxsmall")
    st.subheader("Available tools", divider="gray")
    st.page_link("pages/1_beta.py", label="**Single Asset Beta** — two-asset return regression", icon=":material/trending_up:")
    st.page_link("pages/2_multibeta.py", label="**Beta Tool** — multi-asset regression", icon=":material/line_axis:")
    st.page_link("pages/3_portfoliobeta.py", label="**Portfolio Beta** — weighted portfolio vs benchmark asset(s)", icon=":material/finance:")
    st.page_link("pages/4_hedge.py", label="**Hedge** — beta-weighted hedge construction and backtesting", icon=":material/finance_mode:")
    st.page_link("pages/5_factors.py", label="**Factor Analysis** — equity vs factor regression", icon=":material/analytics:")
    

    st.space('large')
    
    
    # user supplied API key section
    # User supplied API key section
    if "tiingo_key" not in st.session_state:
        st.session_state.tiingo_key = ""

    if "tiingo_key_input" not in st.session_state:
        st.session_state.tiingo_key_input = st.session_state.tiingo_key


    def validate_tiingo_key():
        user_tiingo = st.session_state.tiingo_key_input

        if user_tiingo == st.session_state.tiingo_key:
            return

        if not user_tiingo:
            return

        with st.spinner("Validating key", show_time=True):
            is_valid = is_valid_tiingo_key(user_tiingo)

        if is_valid:
            st.session_state.tiingo_key = user_tiingo
            st.toast(
                "API key saved successfully for this session!",
                icon=":material/verified:",
            )
        else:
            st.toast(
                "Entered API key does not work. Please try again!",
                icon=":material/warning:",
            )
    
    with st.expander("📢 **A note on rate limits**", expanded=False):
        st.markdown("If you are facing issues, you may wish to use your own Tiingo API key for the data. This tool pulls asset data live from Tiingo, and currently it is using my personal API key which may run into rate limits since it is on their free plan. 😿")
        st.link_button("Get your free Tiingo API key", "https://www.tiingo.com/", icon=":material/arrow_outward:", icon_position="right")
        
        
        st.text_input(
            label="Enter your Tiingo API key:",
            type="password",
            key="tiingo_key_input",
            help="Your key remains hidden and is stored only for this session.",
            on_change=validate_tiingo_key,
        )
                
    
    
    
    st.space('xsmall')
    
    
    st.markdown("jubin wang")
    with st.container(width='content', gap='xsmall', horizontal=True):
        st.link_button("GitHub", "https://github.com/wangjubin-06", type='secondary', width='stretch')
        st.link_button("LinkedIn", "https://linkedin.com/in/jubin-wang", type='secondary', width='stretch')
        #st.link_button("Instagram", "https://instagram.com/jubin.w", type='secondary', width='stretch')