import streamlit as st


home = st.Page("pages/home.py", title="Overview", icon="😊", default=True)
beta = st.Page("pages/1_beta.py", title="Beta")
multibeta = st.Page("pages/2_multibeta.py", title="MultiBeta")
portfolio_beta = st.Page("pages/3_portfoliobeta.py", title="Portfolio Beta")
hedge = st.Page("pages/4_hedge.py", title="Hedge")
factors = st.Page("pages/5_factors.py", title="Factors")

pg = st.navigation([home, beta, multibeta, portfolio_beta, hedge, factors])
pg.run()