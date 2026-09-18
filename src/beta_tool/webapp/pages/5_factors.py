import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from beta_tool.factor_research.factorsregression import EquityFactorsRegression
from beta_tool.webapp.tickers import get_tickers, find_tickers


st.set_page_config(
    page_title="Factor Analysis",
    page_icon="🔥",
    layout="wide",
)

st.title("Asset Factor Analysis Tool")

st.markdown(
    """
    Calculate the betas (factor loadings) of an asset's returns against Fama-French Factors
    using ordinary least squares (OLS) regression. This does regression with $$y = \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n +  \\alpha + \\epsilon $$ 
    
    You can also choose the return frequency, return methodology,
    observation period, and HAC-aware standard errors.
    """
)

st.markdown("")

with st.container(border=True):
    st.subheader("Regression inputs")

    # Session state
    if "factors_assets" not in st.session_state:
        st.session_state.factors_assets = []

    # Search
    search_query = st.text_input("Search for tickers", placeholder="enter 2 characters to start", key='multibeta_search_box')

    dropdown_options = find_tickers(st.session_state.ticker_list,search_query,limit=20)


    # assets tickers
    options_pool = dropdown_options.copy()

    # Keep all currently selected X tickers available
    for ticker in st.session_state.factors_assets:
        if ticker not in options_pool:
            options_pool.append(ticker)


    assets = st.multiselect(
        "Ticker(s):",
        options=options_pool,
        key='factors_assets',
        placeholder="select one or many tickers from search"
    )
    


    with st.form("factors_form", border=False, enter_to_submit=False):
            
        col1, col2 = st.columns(2)

        with col1:
            frequency = st.radio(
                "Choose frequency of data",
                options=["daily", "weekly", "monthly"], key="factors_frequency", index=0, horizontal=True
            )
        with col2:
            return_type = st.radio(
                "Choose how returns are calculated",
                options=["log", "simple"], key="factors_return_type", index=1, horizontal=True
            )


        col1, col2 = st.columns(2)
        with col1:
            factor_source = st.radio(
                "Choose factor data (Note that ETF proxy is not the true FF 5-factors)",
                options=["french","etf"], key="factors_factor_source", index=0, horizontal=True,
            )
            

        st.markdown("**Custom date range**")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start date for regression", value=None, key="factors_start_date")
        with col2:
            end_date = st.date_input("End date for regression", value="today", key="factors_end_date")

        st.markdown("")

        st.markdown("### Regression errors")

        col1, col2 = st.columns(2)

        with col1:
            hac = st.radio(
                "Use HAC-aware standard errors?",
                options=[True, False],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                index=1,
                key='factors_hac'
            )

        with col2:
            hac_lag = st.number_input(
                "HAC lags (optional)",
                min_value=1,
                max_value=40,
                value=False,
                step=1,
                key='factors_hac_lag'
            )


        st.markdown("")
        st.markdown("")

        submitted = st.form_submit_button(
            "Run Analysis",
            type="primary",
            use_container_width=True,
        )


if submitted:

    # Basic validation
    if not assets:
        st.error("Please enter asset ticker field.")

    
    elif (start_date is None) and (end_date is None):
        st.error("Please input start date and end date.")

    elif (start_date > end_date):
        st.error("Start date must be earlier than end date.")


    else:
        # Convert dates
        start_date_str = (
            start_date.strftime("%Y-%m-%d")
        )

        end_date_str = (
            end_date.strftime("%Y-%m-%d")
        )



        with st.spinner("Fetching data and running regression..."):

            try:
                if hac:
                    if not hac_lag:
                        beta_obj = EquityFactorsRegression(
                            factor_source=factor_source,
                            frequency=frequency,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            return_type=return_type,
                            hac="auto",
                        )
                    elif hac_lag is not None:
                        beta_obj = EquityFactorsRegression(
                            factor_source=factor_source,
                            frequency=frequency,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            return_type=return_type,
                            hac=hac_lag,
                        )
                elif not hac:
                    beta_obj = EquityFactorsRegression(
                        factor_source=factor_source,
                        frequency=frequency,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        return_type=return_type,
                        hac=None,
                    )

                beta_obj.asset_list(*assets)

                beta_obj.regress()

                # Results dict
                results_dic = beta_obj._grand_results


                # Returns dict
                returns_dic = beta_obj.merged_df_dic


                # Stash everything the display section needs
                st.session_state["factors_results"] = {
                    "results_dic": results_dic,
                    "returns_dic": returns_dic,
                    "assets": assets,
                }


            except Exception as e:
                st.error(f"Regression failed: {e}")
                st.session_state.pop("results", None)
                st.stop()
                time.sleep(5)
                st.rerun()


if 'factors_results' in st.session_state:
    r = st.session_state["factors_results"]
    results_dic, returns_dic = r["results_dic"], r["returns_dic"]
    assets = r["assets"]

    st.subheader('Results')

    st.json(results_dic)

    st.json(returns_dic)
