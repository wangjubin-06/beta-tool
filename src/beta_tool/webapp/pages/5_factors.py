import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from beta_tool.factor_research.factorsregression import EquityFactorsRegression
from beta_tool.factor_research.factorresults import interpret_factor
from beta_tool.webapp.tickers import get_tickers, find_tickers
import time
import json
from beta_tool.data_collection.tiingo_api import tiingo_key_override


st.set_page_config(
    page_title="Factor Analysis",
    page_icon=":material/analytics:",
    layout="wide",
)


st.title("Asset Factor Analysis Tool")
st.space("small")
st.markdown(
    """
    Calculate the betas (factor loadings) of an asset's returns against Fama-French Factors
    using ordinary least squares (OLS) regression. This does regression with $$y = \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n +  \\alpha + \\epsilon $$ 
    
    You can also choose the return frequency, return methodology,
    observation period, and HAC-aware standard errors.
    """
)

st.space("small")

with st.container(border=True):
    st.subheader("Regression inputs")

    # Session state
    if "factors_assets" not in st.session_state:
        st.session_state.factors_assets = []

    # Search
    search_query = st.text_input("Search for tickers", placeholder="enter 1 character to start", key='factors_search_box', icon="🔍")

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
        placeholder="select one or many tickers from search",
        persist_state='session',
    )
    


    with st.form("factors_form", border=False, enter_to_submit=False):
            
        col1, col2 = st.columns(2)

        with col1:
            frequency = st.radio(
                "Choose frequency of data",
                options=["daily", "monthly", "annually"],
                key="factors_frequency",
                index=0,
                horizontal=True,
                persist_state='session'
            )
        with col2:
            return_type = st.radio(
                "Choose how returns are calculated",
                options=["log", "simple"],
                key="factors_return_type",
                index=1,
                horizontal=True,
                persist_state='session',
            )


        col1, col2 = st.columns(2)
        with col1:
            factor_source = st.radio(
                "Choose factor data (Note that ETF proxy is not the true FF 5-factors)",
                options=["french","etf"],
                key="factors_factor_source",
                index=0,
                horizontal=True,
                persist_state='session'
            )
            

        st.markdown("**Custom date range**")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start date for regression", value= None, key="factors_start_date", persist_state='session')
        with col2:
            end_date = st.date_input("End date for regression", value="today", key="factors_end_date", persist_state='session')

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
                key='factors_hac',
                persist_state='session'
            )

        
        with col2:
            hac_lag = st.number_input(
                "HAC lags (optional)",
                min_value=1,
                max_value=40,
                value=None,
                step=1,
                key='factors_hac_lag',
                persist_state='session'
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

    
    elif (start_date is None) or (end_date is None):
        st.error("Please input start date and/or end date.")

    elif (start_date > end_date):
        st.error("Start date must be earlier than end date.")

    elif (
        start_date == end_date
    ):
        st.error("End date cannot be same as start date.")

    else:
        # Convert dates
        start_date_str = (
            start_date.strftime("%Y-%m-%d")
        )

        end_date_str = (
            end_date.strftime("%Y-%m-%d")
        )




        
            
            
        try:
            with st.spinner("Fetching data and running regression...", show_time=True):
                
                if not hac:
                    with tiingo_key_override(st.session_state.get("tiingo_key")):
                        beta_obj = EquityFactorsRegression(
                            factor_source=factor_source,
                            frequency=frequency,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            return_type=return_type,
                        )
                elif hac:
                    if hac_lag is None:
                        hac_input = 'auto'
                    elif isinstance(hac_lag, int) and hac_lag > 0:
                        hac_input = hac_lag
                
                    with tiingo_key_override(st.session_state.get("tiingo_key")):
                        beta_obj = EquityFactorsRegression(
                            factor_source=factor_source,
                            frequency=frequency,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            return_type=return_type,
                            hac=hac_input
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
                    "factor_source": factor_source,
                }


        except Exception as e:
            st.error(f"Regression failed: {e}")
            st.session_state.pop("factors_results", None)
            if st.button("Try again"):
                st.rerun()


if 'factors_results' in st.session_state:
    r = st.session_state["factors_results"]
    results_dic, returns_dic = r["results_dic"], r["returns_dic"]
    assets = r["assets"]
    
    st.space('medium')
    
    if r["factor_source"] == 'french':
        
        st.subheader(f"Fama-French 5 Factors Regression Results")
        
    elif r["factor_source"] == 'etf':
        
        st.subheader(f"ETF-proxy Factors Regression Results")
    
    st.markdown(":small[Due to data availability constraints, the dates may not match your chosen start and end dates :(]")
    
    for ticker, dict in results_dic.items():
        
    
        st.markdown(f"##### {ticker.upper()} Factors")
        
        
        with st.expander(f"Details", expanded=True):
            
            col1, col2 = st.columns(2)
            col1.markdown(f"Start date: {dict['model']["start_date"]}")
            col2.markdown(f"End date: {dict['model']["end_date"]}")
            
            col1, col2 = st.columns(2)
            
            col1.metric("R²", f"{float(dict['model']["r_squared"]):.3f}", border=True)
            col2.metric("N Obs", f"{int(dict['model']["n_observations"])}", border=True)
            
            
            
            col1, col2, col3, col4 = st.columns(4)
            for exposure, exp_dic in dict['exposures'].items():
                beta_val = float(exp_dic['beta'])
                pval = float(exp_dic['p-value'])
                
                interpretation_str = interpret_factor(factor_name=exposure,beta=beta_val, p_value=pval)
                
                
                # Raw results metrics
                with st.container():
                    st.markdown(f"##### {exposure.title()} Factor")
                    st.caption(interpretation_str)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Beta", f"{beta_val:.3f}", border=True)     
                    col2.metric("P-value", f"{float(pval):.2e}", border=True, height='stretch')
    
    st.space('xxsmall')
    #st.json(results_dic)
    st.markdown("##### Full JSON data")
    with st.expander("Full JSON data"):
        
        with st.expander("JSON code", expanded=True):
            st.code(json.dumps(results_dic, indent=4), language="json")
        
        with st.container(horizontal_alignment='right'):
            st.download_button(
                label="Download JSON",
                data=json.dumps(results_dic, indent=4),
                file_name=f"full_factors_data.json",
                mime="application/json",
                key=f"full-json-btn"
            )
        

    #st.json(returns_dic)
    st.space('small')
    
    st.markdown("##### Raw Returns Data")
    
    for ticker, df in returns_dic.items():
        with st.expander(label=f"{ticker.upper()} returns data"):
            st.dataframe(data=df.copy(), hide_index=True)
            
            
            
            with st.container(horizontal=True, horizontal_alignment='right'):
                st.download_button("Download CSV", df.to_csv(index=False), f"{ticker}_data.csv", key=f"{ticker}-csv-btn")

                df_js_str = df.to_json(orient="records", indent=4,date_format='iso')

                
                st.download_button(
                    label="Download JSON",
                    data=df_js_str,
                    file_name=f"{ticker}_data.json",
                    mime="application/json",
                    key=f"{ticker}-json-btn"
                )
    
    st.space('medium')
    # Reset button
    if st.button("Reset Regression"):
        keys_to_clear = [
            "factors_factor_source", "factors_assets", "factors_frequency", "factors_return_type",
            "factors_start_date", "factors_end_date", "factors_hac", "factors_hac_lag", "factors_results",'factors_search_box',
            "factors_form"
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)

        
        st.rerun()


