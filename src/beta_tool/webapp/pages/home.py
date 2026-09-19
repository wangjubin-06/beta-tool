import streamlit as st


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
    st.markdown("""
    - **Single Asset Beta** — two-asset return regression
    - **Beta Tool** — multi-asset regression
    - **Portfolio Beta** — weighted portfolio vs benchmark asset(s)
    - **Hedge** — beta-weighted hedge construction and backtesting
    - **Factor Analysis** — equity vs factor regression
    """)

    st.space('medium')
    
    
    if "tiingo_key" not in st.session_state:
        st.session_state.tiingo_key = ""
    
    st.markdown("If you are facing issues, you may wish to use your own Tiingo API key for the data. This tool pulls asset data live from Tiingo, and currently it is using my personal API key which may run into rate limits since it is on their free plan. 😿")
    st.link_button("Get free Tiingo API key", "https://www.tiingo.com/")
    user_tiingo = st.text_input(
        label='enter your Tiingo API key',
        type='password',
        value= st.session_state.tiingo_key,
        help="Your key remains hidden and is stored only for this session.",
    )
    
    if user_tiingo:
        st.session_state.tiingo_key = user_tiingo
        
        if st.session_state.tiingo_key:
            st.success("API key saved successfully for this session!")
        else:
            st.warning("API key not captured. Please try again!")
    
    
    st.space("large")

    with st.container(width='content', gap="xxsmall"):
        st.markdown("😎😂🤣🤨😐😶‍🌫️😮😪🫩😭😨🥵🤪🤕🤡👹👺😈💩😹😻😿🧟‍♂️👨‍🌾🙇‍♂️🤦‍♂️")
        st.markdown("Made by Jubin Wang")

    st.markdown("Socials:")
    with st.container(width='content', gap='xsmall', horizontal=True):
        st.link_button("GitHub", "https://github.com/wangjubin-06", type='primary', width='stretch')
        st.link_button("LinkedIn", "https://linkedin.com/in/jubin-wang", type='primary', width='stretch')
        st.link_button("Instagram", "https://instagram.com/jubin.w", type='primary', width='stretch')