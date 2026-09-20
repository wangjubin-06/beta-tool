import streamlit as st
import time
from beta_tool.webapp.tickers import is_valid_tiingo_key




st.set_page_config(
    page_title="Home",
    page_icon="🚀", # Can be an emoji or a path to an image file
    layout="centered"
)



with st.container(height='content'):
    st.title("beta_tool")
    st.markdown(
        "##### :rainbow[A quantitative finance toolkit for beta estimation, portfolio hedging, "
        "and factor analysis. Select a tool from the sidebar.]"
    )
    st.space("xxsmall")
    st.subheader("Available tools", divider="gray")
    st.page_link("pages/1_beta.py", label="**Single Asset Beta** — two-asset return regression", icon=":material/trending_up:")
    st.page_link("pages/2_multibeta.py", label="**Beta Tool** — multi-asset regression", icon=":material/line_axis:")
    st.page_link("pages/3_portfoliobeta.py", label="**Portfolio Beta** — weighted portfolio vs benchmark asset(s)", icon=":material/finance:")
    st.page_link("pages/4_hedge.py", label="**Hedge** — beta-weighted hedge construction and backtesting", icon=":material/finance_mode:")
    st.page_link("pages/5_factors.py", label="**Factor Analysis** — equity vs factor regression", icon=":material/analytics:")
    
    # st.markdown("""
    # - **Single Asset Beta** — two-asset return regression
    # - **Beta Tool** — multi-asset regression
    # - **Portfolio Beta** — weighted portfolio vs benchmark asset(s)
    # - **Hedge** — beta-weighted hedge construction and backtesting
    # - **Factor Analysis** — equity vs factor regression
    # """)

    st.space('large')
    
    
    
    # User supplied API key section
    
    if "tiingo_key" not in st.session_state:
        st.session_state.tiingo_key = ""
    

    with st.expander("📢 **A note on rate limits**", expanded=False):
        st.markdown("If you are facing issues, you may wish to use your own Tiingo API key for the data. This tool pulls asset data live from Tiingo, and currently it is using my personal API key which may run into rate limits since it is on their free plan. 😿")
        st.link_button("Get your free Tiingo API key", "https://www.tiingo.com/", icon=":material/arrow_outward:", icon_position="right")
        user_tiingo = st.text_input(
            label='Enter your Tiingo API key:',
            type='password',
            value= st.session_state.tiingo_key,
            help="Your key remains hidden and is stored only for this session.",
        )
    
        if user_tiingo:
            with st.spinner("Validating key", show_time=True):
                is_valid = is_valid_tiingo_key(user_tiingo)
                
            if is_valid:
                st.session_state.tiingo_key = user_tiingo
                success_placeholder = st.empty()
                success_placeholder.success("API key saved successfully for this session!", icon=":material/verified:")
                time.sleep(3)
                success_placeholder.empty()
            else:
                warning_placeholder = st.empty()
                warning_placeholder.warning("Entered API key does not work. Please try again!", icon=":material/warning:")
                time.sleep(3)
                warning_placeholder.empty()
                
    
    
    
    st.space('xsmall')
    
    #st.markdown("😎😂🤣🤨😐😶‍🌫️😮😪🫩😭😨🥵🤪🤕🤡👹👺😈💩😹😻😿🧟‍♂️👨‍🌾🙇‍♂️🤦‍♂️")
    st.markdown("jubin wang")
    with st.container(width='content', gap='xsmall', horizontal=True):
        st.link_button("GitHub", "https://github.com/wangjubin-06", type='secondary', width='stretch')
        st.link_button("LinkedIn", "https://linkedin.com/in/jubin-wang", type='secondary', width='stretch')
        st.link_button("Instagram", "https://instagram.com/jubin.w", type='secondary', width='stretch')