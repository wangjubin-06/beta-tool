import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import time
from beta_tool.regression_beta.portfoliobeta import PortfolioBeta
from beta_tool.webapp.tickers import find_tickers


st.set_page_config(
    page_title="Portfolio Beta",
    layout="wide",
    page_icon="😈"
)



st.title("Portfolio Beta Tool")

st.markdown(
    """
    Calculate the beta of a portfolio's returns against **one** or **multiple** other assets
    using ordinary least squares (OLS) regression. This does regression with $$y = \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n +  \\alpha + \\epsilon $$ 
    
    You can also choose the return frequency, return methodology,
    observation period, and HAC-aware standard errors.
    """
)

st.space("xsmall")


with st.container(border=True):
    st.subheader("Regression inputs")

    # Session states
    if "portfoliobeta_assets" not in st.session_state:
        st.session_state.portfoliobeta_assets = []

    if "portfoliobeta_portdf" not in st.session_state:
        st.session_state.portfoliobeta_portdf = pd.DataFrame({"ticker": pd.Series(dtype="str"),"weight": pd.Series(dtype="float64"),})


    # Search
    search_query = st.text_input("Search for tickers", placeholder="enter 2 characters to start", key='portfoliobeta_search_box', icon="🔍")

    dropdown_options = find_tickers(st.session_state.ticker_list,search_query,limit=20)


    # Y-ticker
    options_pool_y = dropdown_options.copy()

    selected_port_tickers = (
        st.session_state.portfoliobeta_portdf["ticker"].dropna().tolist()
    )
    
    for ticker in selected_port_tickers:
        if ticker not in options_pool_y:
            options_pool_y.append(ticker)

    # X-tickers
    options_pool_x = dropdown_options.copy()

    # Keep all currently selected X tickers available
    for ticker in st.session_state.portfoliobeta_assets:
        if ticker not in options_pool_x:
            options_pool_x.append(ticker)


    with st.container(border=True):
        
        st.write("Portfolio holdings:")

        add_col1, add_col2, add_col3 = st.columns([3, 2, 1], width='stretch')
        with add_col1:
            staged_ticker = st.selectbox(
                "Select ticker to add",
                options=dropdown_options,
                key="portfoliobeta_staged_ticker",
                label_visibility="collapsed",
                placeholder="pick from search results",
            )
        with add_col2:
            staged_weight = st.number_input(
                label="Weight in %", min_value=0.01, max_value=100.0, step=0.1,
                key="portfoliobeta_staged_weight", label_visibility="collapsed",
            )
        with add_col3:
            add_clicked = st.button("Add to portfolio", key="portfoliobeta_add_btn", width='stretch')


        

        if add_clicked:
            current_total = st.session_state.portfoliobeta_portdf["weight"].sum()
            if not staged_ticker:
                st.warning("Select a ticker to add.")
            elif staged_ticker in st.session_state.portfoliobeta_portdf["ticker"].values:
                st.warning(f"{staged_ticker} is already in the portfolio.")
            elif current_total + staged_weight > 100 + 1e-6:
                st.error(
                    f"Adding {staged_weight:.3f}% would bring the total to "
                    f"{current_total + staged_weight:.3f}%, over the 100% limit. "
                    f"You have {100 - current_total:.3f}% remaining."
                )
            else:
                new_row = pd.DataFrame({"ticker": [staged_ticker], "weight": [staged_weight]})
                st.session_state.portfoliobeta_portdf = pd.concat(
                    [st.session_state.portfoliobeta_portdf, new_row], ignore_index=True
                )



        st.space("xsmall")


        # options = full static ticker_list, NOT the live search results —
        # this is what stops the column_config from changing shape every rerun
        edited_df = st.data_editor(
            data=st.session_state.portfoliobeta_portdf.copy(),
            hide_index=True,
            num_rows="fixed",
            column_config={
                "ticker": st.column_config.SelectboxColumn("Ticker", required=True, options=options_pool_y),
                "weight": st.column_config.NumberColumn(
                    "Weight (%)", min_value=0.0, max_value=100.0, step=0.001, format="%.3f"
                ),
            },
            key="portfoliobeta_port_editor",
            width="stretch",
        )


        st.session_state.portfoliobeta_portdf = edited_df.copy()

        st.markdown("")

        remove_col1, remove_col2 = st.columns([5,1], width='stretch')
        with remove_col1:
            ticker_to_remove = st.selectbox(
                "Remove a ticker",
                options=st.session_state.portfoliobeta_portdf["ticker"].dropna().tolist(),
                key="portfoliobeta_remove_ticker",
                label_visibility="collapsed",
                placeholder="select a ticker to remove",
            )
        with remove_col2:
            if st.button("Remove", key="portfoliobeta_remove_btn", width='stretch'):
                if ticker_to_remove:
                    st.session_state.portfoliobeta_portdf = st.session_state.portfoliobeta_portdf[
                        st.session_state.portfoliobeta_portdf["ticker"] != ticker_to_remove
                    ].reset_index(drop=True)
                    st.rerun()

        


    with st.container(border=True):
        st.write("Independent (x) ticker(s):")
        assets = st.multiselect(
            label="",
            label_visibility='collapsed',
            options=options_pool_x,
            key='portfoliobeta_assets',
            placeholder="select one or many tickers from search"
        )

    # Rest of form
    with st.form("portfoliobeta_form", border=False, enter_to_submit=False):
        st.space('xsmall')
        col1, col2 = st.columns(2)

        with col1:
            frequency = st.radio(
                "Choose frequency of data",
                options=["daily", "weekly", "monthly"], key="portfoliobeta_frequency", index=0, horizontal=True
            )
        with col2:
            return_type = st.radio(
                "Choose how returns are calculated",
                options=["log", "simple"], key="portfoliobeta_return_type", index=1, horizontal=True
            )


        

        period = st.selectbox("Regression period", options=['1m','3m','6m','1y','2y','3y','5y','10y','20y','30y'], index=3, key='portfoliobeta_period')
        
            
        st.space('xsmall')
        st.markdown("**Custom date range**")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start date for regression (optional)", value=None, key="portfoliobeta_start_date")
        with col2:
            end_date = st.date_input("End date for regression (optional)", value=None, key="portfoliobeta_end_date")

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
                key='portfoliobeta_hac'
            )

        with col2:
            hac_lag = st.number_input(
                "HAC lags",
                min_value=1,
                max_value=40,
                value=5,
                step=1,
                key='portfoliobeta_hac_lag'
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
                key="portfoliobeta_rolling"
            )

        with col2:
            rolling_window = st.number_input(
                "Lookback window for rolling Beta",
                min_value=2,
                max_value=200,
                value=60,
                step=1,
                key="portfoliobeta_rolling_window"
            )

        st.space('small')

        submitted = st.form_submit_button(
            "Run Regression",
            type="primary",
            use_container_width=True,
        )


if submitted:

    # Converting edited_df to a dictionary

    port_df = edited_df.copy()

    port_dict = dict(zip(port_df['ticker'], port_df['weight']))

    

    # Basic validation
    if len(port_dict) == 0:
        st.error("Please enter at least 1 ticker in portfolio")

    
    if not assets:
        st.error("Please enter (y) assets ticker field.")
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

        if len(assets) == 1:
            assets = assets[0]


        # -------------------------------------------------
        # Run regression
        # -------------------------------------------------

        try:
        
            with st.spinner("Fetching data and running regression..."):
                beta_obj = PortfolioBeta(
                    portfolio_dic=port_dict,
                    asset_to_be_regressed=assets,
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
                portfolio_returns_df = beta_obj.portfolio_returns_data.copy()

                independent_returns_data = beta_obj.independent_data

                independent_returns_dic = {}

                if isinstance(independent_returns_data, dict):
                    for ticker, df in independent_returns_data.items():
                        independent_returns_dic[ticker.upper()] = df.copy()
                else:
                    independent_returns_data = independent_returns_data.copy()
                    independent_returns_data.rename(columns={"independent_variable_returns": f"{ticker.upper()}-returns"}, inplace=True)


                

                for ticker, df in independent_returns_dic.items():
                    df.rename(columns={f"{return_type}-returns": f"{ticker}-returns"}, inplace=True)

                if isinstance(independent_returns_data,dict):
                    independent_returns_data = independent_returns_dic



                # Stash everything the display section needs
                st.session_state["portfoliobeta_results"] = {
                    "results_dic": results_dic,
                    "port_returns_df": portfolio_returns_df,
                    "ind_returns_df": independent_returns_data,
                    "port_dict": port_dict,
                    "assets": assets,
                    "return_type": return_type,
                }

                if isinstance(independent_returns_data,dict):
                    st.session_state["portfoliobeta_results"]["x_col"] = beta_obj.merged_assets_names.copy()
                else:
                    st.session_state["portfoliobeta_results"]["x_col"] = [assets,]

                if rolling:

                    rolling_df_data = beta_obj.historical_rolling_beta(window=rolling_window)

                    
                    st.session_state["portfoliobeta_results"]["rolling_dfs"] = rolling_df_data

                 

        except Exception as e:
            st.error(f"Regression failed: {e}")
            st.session_state.pop("portfoliobeta_results", None)
            # time.sleep(5)
            # st.rerun()
            if st.button("try again", icon="😭"):
                st.rerun()


if 'portfoliobeta_results' in st.session_state:
    r = st.session_state["portfoliobeta_results"]
    r_dic = r["results_dic"]
    port_returns_df = r["port_returns_df"]
    ind_returns_df = r["ind_returns_df"]
    port_dict = r["port_dict"]
    assets = r["assets"]
    return_type = r['return_type']
    x_col = r['x_col']

    if rolling:
        rolling_df_data = r["rolling_dfs"]

    st.space('small')
    
    st.subheader('Results')
    
    
    col1, col2 = st.columns(2)
    
    col1.metric("R²", f"{float(r_dic['r_squared']):.3f}", border=True)
    
    col2.metric("N Obs", f"{int(r_dic["n_obs"])}", border=True)



    for col in x_col:
        
        if len(x_col) > 1:
            
            st.markdown(f"##### Beta - {col.upper()}")
            
            beta_val = float(r_dic[col.lower()]['beta'])
            
            pval = float(r_dic[col.lower()]['beta_pvalue'])
            sig = "statistically significant" if pval < 0.05 else "**not** statistically significant at the 5% level"
            
            st.caption(
                f"A 1% move in {col.upper()} is associated with a {beta_val:.2f}% move in portfolio, "
                f"on average ({sig}), keeping all other assets constant."
            )

            st.space("xsmall")
            
            # Raw results metrics
            col1, col2 = st.columns(2)
            col1.metric("Beta", f"{beta_val:.3f}", delta=f"{beta_val - 1:.3f} vs 1.0", delta_color="off", border=True)     
            col2.metric("P-value", f"{float(r_dic[col]["beta_pvalue"]):.2e}", border=True, height='stretch')

        elif len(x_col) == 1:
            st.markdown(f"##### Beta - {col.upper()}")
            beta_val = float(r_dic['beta'])
            pval = float(r_dic['beta_pvalue'])
            sig = "statistically significant" if pval < 0.05 else "**not** statistically significant at the 5% level"
            st.caption(
                f"A 1% move in {col.upper()} is associated with a {beta_val:.2f}% move in portfolio, "
                f"on average ({sig}), keeping all other assets constant."
            )

            st.space("xsmall")
            # Raw results metrics
            col1, col2 = st.columns(2)
            col1.metric("Beta", f"{beta_val:.3f}", delta=f"{beta_val - 1:.3f} vs 1.0", delta_color="off", border=True)     
            col2.metric("P-value", f"{float(r_dic["beta_pvalue"]):.2e}", border=True, height='stretch')



    st.markdown("")
    st.markdown("")

    with st.expander("Full regression stats"):
        st.markdown(f"###### Basic stats")
        basic_stats = dict(list(r_dic.items())[:12])
        basics_df = pd.DataFrame(basic_stats, index=[0])
        st.dataframe(basics_df, hide_index=True)

        
        with st.container(horizontal=True, horizontal_alignment='right'):
            st.download_button("Download CSV", basics_df.to_csv(index=False), "basic_regression_stats.csv", key="portfoliobeta_basic_csv_download")

            df_js_str = basics_df.to_json(orient="records", indent=4,date_format='iso')

        
            st.download_button(
                label="Download JSON",
                data=df_js_str,
                file_name="basic_regression_stats.json",
                mime="application/json",
                key="portfoliobeta_basic_json_download"
            )

        if len(x_col) > 1:
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
        elif len(x_col) == 1:
            
            st.markdown(f"###### Beta w.r.t {x_col[0].upper()}")
            df = pd.DataFrame(r_dic, index=[0])
            st.dataframe(df,width='stretch', hide_index=True)

            
            with st.container(horizontal=True, horizontal_alignment='right'):
                st.download_button("Download CSV", df.to_csv(index=False), f"{x_col[0]}_regression_stats.csv", key=f"{x_col[0]}-csv-download")

                df_js_str = df.to_json(orient="records", indent=4,date_format='iso')

            
                st.download_button(
                    label="Download JSON",
                    data=df_js_str,
                    file_name=f"{x_col[0]}_regression_stats.json",
                    mime="application/json",
                    key=f"{x_col[0]}-json-download"
                )




    # Portfolio cumulative returns plot
    st.space("small")
    cum_df = port_returns_df.set_index("date")[[f'portfolio_{return_type}_returns']]
    
    if return_type == "log":
        cum_df = np.exp(cum_df.cumsum()) - 1
    else:
        cum_df = (1 + cum_df).cumprod() - 1

    fig2 = go.Figure()
    #for col, color in zip(cum_df.columns, ["#4C78A8", "#E45756"]):
    fig2.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df[f'portfolio_{return_type}_returns'], mode="lines", name=f'Portfolio returns',
        line=dict(width=2),
    ))

    if len(x_col) > 1:
        for col in x_col:
            df = ind_returns_df[col.upper()]
            cum_df = df.set_index("date")[[f'{col.upper()}-returns']]

            if return_type == "log":
                cum_df = np.exp(cum_df.cumsum()) - 1
            else:
                cum_df = (1 + cum_df).cumprod() - 1

            fig2.add_trace(go.Scatter(
                x = cum_df.index, y=cum_df[f'{col.upper()}-returns'], mode="lines", name=f'{col.upper()} returns',
                line=dict(width=2),
            )
            )
    elif len(x_col) == 1:
        df = ind_returns_df.copy()
        cum_df = df.set_index("date")[[f'{x_col[0].upper()}-returns']]

        if return_type == "log":
            cum_df = np.exp(cum_df.cumsum()) - 1
        else:
            cum_df = (1 + cum_df).cumprod() - 1

        fig2.add_trace(go.Scatter(
            x = cum_df.index, y=cum_df[f'{x_col[0].upper()}-returns'], mode="lines", name=f'{x_col[0].upper()} returns',
            line=dict(width=2),
        )
        )
        
    fig2.update_layout(
        #title="Cumulative Returns",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, l=10, r=10, b=10),
        hovermode="x unified",
    )
    st.space("small")
    st.markdown('##### Cumulative Returns')
    
    st.plotly_chart(fig2, width='stretch')


    st.space('small')


    # Returns data
    with st.expander("Raw returns data"):
        st.dataframe(port_returns_df, width="stretch", hide_index=True)

        
        with st.container(horizontal=True, horizontal_alignment='right'):
            st.download_button("Download CSV", port_returns_df.to_csv(index=False), "returns.csv", key=f"portfolio-returns-csv-download-btn")
                
            returns_js_str = port_returns_df.to_json(orient="records", indent=4, date_format='iso')

        
            st.download_button(
                label="Download JSON",
                data=returns_js_str,
                file_name="returns.json",
                mime="application/json",
                key="portfolio-returns-json-download-btn"
            )

        st.markdown("")


        if isinstance(ind_returns_df, dict):
            for ticker, df in ind_returns_df.items():
                st.dataframe(df, width='stretch', hide_index=True)

                
                with st.container(horizontal=True, horizontal_alignment='right'):
                    st.download_button("Download CSV", df.to_csv(index=False), "returns.csv", key=f"{ticker}-returns-csv-download-btn")
                        
                    returns_js_str = df.to_json(orient="records", indent=4, date_format='iso')

                
                    st.download_button(
                        label="Download JSON",
                        data=returns_js_str,
                        file_name="returns.json",
                        mime="application/json",
                        key=f"{ticker}-returns-json-download-btn"
                    )

                st.markdown("")
                st.markdown("")

        else:
            st.dataframe(ind_returns_df, width='stretch', hide_index=True)

            
            with st.container(horizontal=True, horizontal_alignment='right'):
                st.download_button("Download CSV", ind_returns_df.to_csv(index=False), "returns.csv")
                    
                returns_js_str = ind_returns_df.to_json(orient="records", indent=4, date_format='iso')

            
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


        #st.write(rolling_df_data)

        if len(x_col) == 1:
            rol_df = rolling_df_data.set_index("date")[["beta","beta_ci_upper","beta_ci_lower"]]

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
                #title=f"{rolling_window}-observations Rolling Beta with 95% Confidence Intervals",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, l=10, r=10, b=10),
                hovermode="x unified",
            )
            st.space("small")
            st.markdown(f"##### {rolling_window}-observations Rolling Beta with 95% Confidence Intervals")
            st.plotly_chart(fig, width='stretch')


            # Rolling beta stats
            st.markdown("")
            with st.expander("Rolling Beta Stats"):    
                df = rolling_df_data[["date","beta","beta_ci_upper","beta_ci_lower"]].copy()
                df.dropna(ignore_index=True, inplace=True)

                st.markdown(f"##### {x_col[0].upper()} Rolling Beta Stats ")
                st.dataframe(df, width='stretch', hide_index=True)

                

                with st.container(horizontal=True, horizontal_alignment='right'):
                    st.download_button("Download CSV", df.to_csv(index=False), f"{ticker}_rolling_stats.csv")

                    rol_js_str = df.to_json(orient="records", indent=4, date_format='iso')

                
                    st.download_button(
                        label="Download JSON",
                        data=rol_js_str,
                        file_name=f"portfolio_{ticker}_{rolling_window}_rolling_beta_stats.json",
                        mime="application/json"
                    )

            st.space("small")


        elif len(x_col) > 1:
            for ticker, df in rolling_df_data.items():
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
                #title=f"{rolling_window}-observations Rolling Beta with 95% Confidence Intervals",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, l=10, r=10, b=10),
                hovermode="x unified",
            )
            st.space("small")
            st.markdown(f"##### {rolling_window}-observations Rolling Beta with 95% Confidence Intervals")
            st.plotly_chart(fig, width='stretch')

            # Rolling beta stats
            st.space("small")
            with st.expander("Rolling Beta Stats"):

                for ticker, df in rolling_df_data.items():
                    
                    df = df.copy()
                    df = df[["date","beta","beta_ci_upper","beta_ci_lower"]]
                    df.dropna(ignore_index=True, inplace=True)

                    st.markdown(f"##### {ticker.upper()} Rolling Beta Stats ")
                    st.dataframe(df, width='stretch', hide_index=True)

                    

                    with st.container(horizontal=True, horizontal_alignment='right'):
                        st.download_button("Download CSV", df.to_csv(index=False), f"{ticker}_rolling_stats.csv")

                        rol_js_str = df.to_json(orient="records", indent=4, date_format='iso')

                    
                        st.download_button(
                            label="Download JSON",
                            data=rol_js_str,
                            file_name=f"portfolio_{ticker}_{rolling_window}_rolling_beta_stats.json",
                            mime="application/json"
                        )

                st.space("xsmall")

    st.space('medium')
    
    # Reset button
    if st.button("Reset Regression"):
        keys_to_clear = [
            "portfoliobeta_portdf", "portfoliobeta_assets", "portfoliobeta_frequency", "portfoliobeta_return_type", "portfoliobeta_period",
            "portfoliobeta_start_date", "portfoliobeta_end_date", "portfoliobeta_hac", "portfoliobeta_hac_lag", "portfoliobeta_results",'portfoliobeta_search_box','portfoliobeta_staged_ticker',
            "portfoliobeta_staged_weight", "portfoliobeta_add_btn","portfoliobeta_port_editor", "portfoliobeta_remove_ticker","portfoliobeta_remove_btn",
            'portfoliobeta_assets',"portfoliobeta_form","portfoliobeta_rolling","portfoliobeta_rolling_window"
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)

        
        st.rerun()


            


    
