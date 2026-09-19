import streamlit as st
import numpy as np
import pandas as pd
import time
from beta_tool.regression_beta.multibeta import MultiBeta
import plotly.express as px
import plotly.graph_objects as go
from beta_tool.webapp.tickers import get_tickers, find_tickers



st.set_page_config(
    page_title="Beta Tool",
    layout="wide",
    page_icon="😻"
)




st.title("Beta Tool")

st.markdown(
    """
    Calculate the beta of one asset's returns against **one** or **multiple** other assets
    using ordinary least squares (OLS) regression. This does regression with $$y = \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n +  \\alpha + \\epsilon $$ 
    
    You can also choose the return frequency, return methodology,
    observation period, and HAC-aware standard errors.
    """
)

st.space("small")


with st.container(border=True):
    st.subheader("Regression inputs")


    # Session state
    if "multibeta_assets" not in st.session_state:
        st.session_state.multibeta_assets = []

    if "multibeta_asset1" not in st.session_state:
        st.session_state.multibeta_asset1 = ""


    # Search
    search_query = st.text_input("Search for tickers", placeholder="enter 2 characters to start", key='multibeta_search_box', icon="🔍")

    dropdown_options = find_tickers(st.session_state.ticker_list,search_query,limit=20)


    # Y-ticker
    options_pool_y = dropdown_options.copy()

    if (
        st.session_state.multibeta_asset1
        and st.session_state.multibeta_asset1 not in options_pool_y
    ):
        options_pool_y.append(st.session_state.multibeta_asset1)


    # X-tickers
    options_pool_x = dropdown_options.copy()

    # Keep all currently selected X tickers available
    for ticker in st.session_state.multibeta_assets:
        if ticker not in options_pool_x:
            options_pool_x.append(ticker)


    # Ticker Entry fields
    col1, col2 = st.columns(2)

    with col1:
        asset1 = st.selectbox(
            "Dependent (y) ticker:",
            options=options_pool_y,
            key='multibeta_asset1',
            placeholder="select a ticker from search"
        )
    with col2:
        assets = st.multiselect(
            "Independent (x) ticker(s):",
            options=options_pool_x,
            key='multibeta_assets',
            placeholder="select one or many tickers from search"
        )


    # Rest of form
    with st.form("multibeta_form", border=False, enter_to_submit=False):
        
        col1, col2 = st.columns(2)

        with col1:
            frequency = st.radio(
                "Choose frequency of data",
                options=["daily", "weekly", "monthly"], key="multibeta_frequency", index=0, horizontal=True
            )
        with col2:
            return_type = st.radio(
                "Choose how returns are calculated",
                options=["log", "simple"], key="multibeta_return_type", index=1, horizontal=True
            )


        col1, col2 = st.columns(2)

        with col1:
            period = st.selectbox("Regression period", options=['1m','3m','6m','1y','2y','3y','5y','10y','20y','30y'], index=3, key='multibeta_period')
        
            

        st.markdown("**Custom date range**")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start date for regression (Optional)", value=None, key="multibeta_start_date")
        with col2:
            end_date = st.date_input("End date for regression (Optional)", value=None, key="multibeta_end_date")

        st.space("xsmall")

        st.markdown("### Regression errors")

        col1, col2 = st.columns(2)

        with col1:
            hac = st.radio(
                "Use HAC-aware standard errors?",
                options=[True, False],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                index=1,
                key='multibeta_hac'
            )

        with col2:
            hac_lag = st.number_input(
                "HAC lags",
                min_value=1,
                max_value=40,
                value=5,
                step=1,
                key='multibeta_hac_lag'
            )

        st.space("xsmall")
        st.markdown("#### Rolling Beta")

        col1, col2 = st.columns(2)

        with col1:
            rolling = st.radio(
                "Do rolling beta?",
                options=[True,False],
                format_func= lambda x: "Yes" if x else "No",
                horizontal=True,
                index=0,
                key="multibeta_rolling"
            )

        with col2:
            rolling_window = st.number_input(
                "Lookback window for rolling Beta",
                min_value=2,
                max_value=200,
                value=60,
                step=1,
                key="multibeta_rolling_window"
            )

        st.markdown("")
        st.markdown("")

        submitted = st.form_submit_button(
            "Run Regression",
            type="primary",
            use_container_width=True,
        )

    
if submitted:

    # Basic validation
    if not asset1 or not assets:
        st.error("Please enter both asset ticker fields.")

    elif (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        st.error("Start date must be earlier than end date.")

    else:
        # Convert dates
        start_date_str = (
            start_date.strftime("%Y-%m-%d")
            if start_date is not None
            else None
        )

        end_date_str = (
            end_date.strftime("%Y-%m-%d")
            if end_date is not None
            else None
        )

        # IMPORTANT:
        # Only pass HAC lag when HAC is enabled.
        selected_hac_lag = hac_lag if hac else None


        # -------------------------------------------------
        # Run regression
        # -------------------------------------------------

        try:
            with st.spinner("Fetching data and running regression..."):

            
                beta_obj = MultiBeta(
                    asset1=asset1,
                    assets=assets,
                    period=period,
                    frequency=frequency,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    return_type=return_type,
                    hac=hac,
                    hac_lag=selected_hac_lag,
                )

                # Results dict
                results_dic = beta_obj.get_static_results()
                

                # Returns dataframe
                y_col = "y"
                x_col = beta_obj.merged_assets_names

                returns_df = beta_obj.merged_return_series.copy()

                returns_df = returns_df.rename(columns={y_col: f"{asset1.upper()}-returns"})

                new_x_col = []

                for col in x_col:
                    returns_df = returns_df.rename(columns={col: f"{col.upper()}-returns"})
                    new_x_col.append(f"{col}-returns")


                y_col = f"{asset1.upper()}-returns"
                


                

                # Stash everything the display section needs
                st.session_state["multibeta_results"] = {
                    "results_dic": results_dic,
                    "returns_df": returns_df,
                    "y_col": y_col,
                    "x_col": x_col,
                    "asset1": asset1,
                    "assets": assets,
                }

                if rolling:
                    rolling_dfs_dic = beta_obj.historical_rolling_beta(window=rolling_window)
                    
                    st.session_state["multibeta_results"]["rolling_dfs_dic"] = rolling_dfs_dic

                

        except Exception as e:
            st.error(f"Regression failed: {e}")
            st.session_state.pop("results", None)
            #st.stop()
            time.sleep(5)
            st.rerun()



if 'multibeta_results' in st.session_state:
    r = st.session_state["multibeta_results"]
    r_dic, returns_df = r["results_dic"], r["returns_df"]
    x_col, y_col = r["x_col"], r["y_col"]
    asset1, assets = r["asset1"], r["assets"]
    

    if rolling:
        roll_dic = r["rolling_dfs_dic"]


    st.subheader('Results')
    col1, col2 = st.columns(2)
    col1.metric("R²", f"{float(r_dic['r_squared']):.3f}", border=True)
    col2.metric("N Obs", f"{int(r_dic["n_obs"])}", border=True)

    # One line summary for each asset

    for col in x_col:
        st.markdown(f"##### Beta - {col}")
        beta_val = float(r_dic[col]['beta'])
        pval = float(r_dic[col]['beta_pvalue'])
        sig = "statistically significant" if pval < 0.05 else "**not** statistically significant at the 5% level"
        st.caption(
            f"A 1% move in {col.upper()} is associated with a {beta_val:.2f}% move in {asset1.upper()}, "
            f"on average ({sig}), keeping all other assets constant."
        )

        st.space('xxsmall')

        # Raw results metrics
        col1, col2 = st.columns(2)
        col1.metric("Beta", f"{beta_val:.3f}", delta=f"{beta_val - 1:.3f} vs 1.0", delta_color="off", border=True)     
        col2.metric("P-value", f"{float(r_dic[col]["beta_pvalue"]):.2e}", border=True, height='stretch')

        

    st.markdown("")
    st.markdown("")

    with st.expander("Full regression stats"):

        st.markdown(f"###### Basic stats")
        basic_stats = dict(list(r_dic.items())[:12])
        basics_df = pd.DataFrame(basic_stats, index=[0])
        st.dataframe(basics_df, hide_index=True)

        
        with st.container(horizontal=True, horizontal_alignment='right'):
            st.download_button("Download CSV", basics_df.to_csv(index=False), "basic_regression_stats.csv", key="multibeta_basic_csv_download")

            df_js_str = basics_df.to_json(orient="records", indent=4,date_format='iso')

        
            st.download_button(
                label="Download JSON",
                data=df_js_str,
                file_name="basic_regression_stats.json",
                mime="application/json",
                key="multibeta_basic_json_download"
            )


        

        for i in range(len(x_col)):
            
            st.markdown(f"###### Beta w.r.t {x_col[i].upper()}")
            df = pd.DataFrame(r_dic[x_col[i]], index=[0])
            st.dataframe(df,width='stretch', hide_index=True)

            
            with st.container(horizontal=True, horizontal_alignment='right'):
                st.download_button("Download CSV", df.to_csv(index=False), f"{x_col[i]}_regression_stats.csv", key=f"{x_col[i]}-csv-download")

                df_js_str = df.to_json(orient="records", indent=4,date_format='iso')

            
                st.download_button(
                    label="Download JSON",
                    data=df_js_str,
                    file_name=f"{x_col[i]}_regression_stats.json",
                    mime="application/json",
                    key=f"{x_col[i]}-json-download"
                )

    st.markdown("")
    # Returns data
    with st.expander("Raw returns data"):
        st.dataframe(returns_df, width='stretch')

        
        with st.container(horizontal=True, horizontal_alignment='right'):
            st.download_button("Download CSV", returns_df.to_csv(index=False), "returns.csv")
                
            returns_js_str = returns_df.to_json(orient="records", indent=4, date_format='iso')

        
            st.download_button(
                label="Download JSON",
                data=returns_js_str,
                file_name="returns.json",
                mime="application/json"
            )


    # Rolling beta data
    st.markdown("")

    if rolling:
        st.markdown("")
        st.markdown("##### Rolling Beta Results")

        st.markdown("")

        # Rolling beta plot
        fig = go.Figure()

        for ticker, df in roll_dic.items():
            df = df.copy()
            df.dropna(ignore_index=True, inplace=True)


            rol_df = df.set_index("date")[["beta","beta_ci_upper","beta_ci_lower"]]

            # Confidence interval
            # 1. Add the lower bound trace (hidden line, used as the fill anchor)
            fig.add_trace(go.Scatter(
                x=rol_df.index,
                y=rol_df['beta_ci_lower'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                #hoverinfo='skip',
                name=f'{ticker} 95% CI lower',
                fillcolor='rgba(0, 0, 0, 0.1)',
            ))

            # 2. Add the upper bound trace and fill down to the lower bound trace
            fig.add_trace(go.Scatter(
                x=rol_df.index,
                y=rol_df['beta_ci_upper'],
                mode='lines',
                line= dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 0, 0, 0.1)',  # Semi-transparent color for the band
                name=f'{ticker} 95% CI upper',
                showlegend=False,
            ))

            # Beta time series
            fig.add_trace(go.Scatter(
                x=rol_df.index, y=rol_df["beta"], mode="lines", name=f"{ticker} beta",
                line=dict(width=2),
            ))

        fig.update_layout(
            title=f"{rolling_window}-observations Rolling Beta with 95% Confidence Intervals",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, l=10, r=10, b=10),
            hovermode="x unified",
        )

        st.plotly_chart(fig, width='stretch')

        # Rolling beta stats
        st.markdown("")
        with st.expander("Rolling Beta Stats"):

            for ticker, df in roll_dic.items():
                
                df = df.copy()
                df.dropna(ignore_index=True, inplace=True)

                st.markdown(f"##### {ticker.upper()} Rolling Beta Stats ")
                st.dataframe(df, width='stretch', hide_index=True)

                

                with st.container(horizontal=True, horizontal_alignment='right'):
                    st.download_button("Download CSV", df.to_csv(index=False), f"{ticker}_rolling_stats.csv")

                    rol_js_str = df.to_json(orient="records", indent=4, date_format='iso')

                
                    st.download_button(
                        label="Download JSON",
                        data=rol_js_str,
                        file_name=f"{asset1}_{ticker}_{rolling_window}_rolling_beta_stats.json",
                        mime="application/json"
                    )

            st.markdown("")

        st.markdown("")
        st.markdown("")

    # Reset button
    if st.button("Reset Regression"):
        keys_to_clear = [
            "multibeta_asset1", "multibeta_assets", "multibeta_frequency", "multibeta_return_type", "multibeta_period",
            "multibeta_start_date", "multibeta_end_date", "multibeta_hac", "multibeta_hac_lag", "multibeta_results",'multibeta_search_box',
            'multibeta_form', 'multibeta_rolling', 'multibeta_rolling_window'
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.rerun()

        
    
