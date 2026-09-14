import streamlit as st
import datetime
#from beta_tool.regression_beta.beta import Beta
#from beta_tool.webapp.plots import rolling_beta_chart


st.title("Single asset beta")
st.write(
    "This tool calculates the beta of an asset's returns against that of another "
    "using standard ordinary least squares regression. It also allows for rolling "
    "beta calculation and visualisation."
)

st.subheader("Enter your inputs")

asset1 = st.text_input("Enter dependent (y) ticker:", placeholder="aapl", key="asset1")
asset2 = st.text_input("Enter independent (x) ticker:", placeholder="spy", key="asset2")


frequency = st.radio(
    "Choose frequency of data",
    options=["daily", "weekly", "monthly"], key="frequency"
)

return_type = st.radio(
    "Choose how returns are calculated",
    options=["log", "simple"], key="return_type"
)

period = st.selectbox("Choose data period for regression", options=['1m','3m','6m','1y','2y','3y','5y','10y','20y','30y'], index=3)

start_date = st.date_input("Pick a start date for regression (Optional)", value=None, key="start_date")

end_date = st.date_input("Pick a end date for regression (Optional)", value=None, key="end_date")

if start_date is not None:
    start_date = start_date.strftime("%Y-%m-%d")

if end_date is not None:
    end_date = start_date.strftime("%Y-%m-%d")


hac = st.radio(
    "Choose whether to use HAC-aware errors in regression",
    options=[True, False], key="hac"
)

hac_lag = st.number_input(
    "Choose lags for HAC-aware regression (optional)",
    min_value=1, max_value=40, value=None, key="hac_lag"
)

