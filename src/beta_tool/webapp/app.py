import streamlit as st
from beta_tool.webapp import theme
from beta_tool.webapp.tickers import get_tickers

theme.apply_theme()

@st.cache_data
def tickers():
    return get_tickers()



home = st.Page("pages/home.py", title="Home", default=True)
beta = st.Page("pages/1_beta.py", title="Single Asset Beta")
multibeta = st.Page("pages/2_multibeta.py", title="Beta Tool")
portfolio_beta = st.Page("pages/3_portfoliobeta.py", title="Portfolio Beta")
hedge = st.Page("pages/4_hedge.py", title="Hedge")
factors = st.Page("pages/5_factors.py", title="Factor Analysis")

pg = st.navigation([home, beta, multibeta, portfolio_beta, hedge, factors])

try:
    tickers_df, ticker_list = tickers()

    if 'ticker_list' not in st.session_state:
        st.session_state.ticker_list = ticker_list

except ConnectionError:
    pg.markdown("Error: device is not connected to internet! Please try again with an internet connection")



pg.run()