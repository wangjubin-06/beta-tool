import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from beta_tool.portfolio_hedge.portfolio_hedger import PortfolioHedge
from beta_tool.webapp.tickers import find_tickers

WINDOW_DEFAULTS = {'daily': 126, 'weekly': 52, 'monthly': 24}
WINDOW_FLOORS   = {'daily': 30,  'weekly': 20, 'monthly': 12}


st.set_page_config(
    page_title="Portfolio Beta",
    page_icon="💹",
    layout="wide",
)



st.title("Portfolio Hedger Tool")

st.markdown(
    """
    This tool hedges a portfolio by shorting a mix of assets, based on the beta of the assets with the portfolio. The effectiveness of the hedge can be visualised.
    You can choose the backtest period to test the effectiveness across different market regimes. You can also choose between a static hedge or a rolling hedge which
    rebalances the hedging assets with your portfolio periodically based on recalculated rolling betas.
    
    This does regression with $$y = \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n +  \\alpha + \\epsilon $$ 
    
    You can also choose the return frequency, return methodology, observation period, and HAC-aware standard errors.
    """
)

st.space("xsmall")


with st.container(border=True):
    st.subheader("Regression inputs")
    
    # Session states
    if "hedge_assets" not in st.session_state:
        st.session_state.hedge_assets = []

    if "hedge_portdf" not in st.session_state:
        st.session_state.hedge_portdf = pd.DataFrame({"ticker": pd.Series(dtype="str"),"weight": pd.Series(dtype="float64"),})
        

    # Search
    search_query = st.text_input("Search for tickers", placeholder="enter 2 characters to start", key='hedge_search_box', icon="🔍")

    dropdown_options = find_tickers(st.session_state.ticker_list,search_query,limit=20)


    # Y-ticker
    options_pool_y = dropdown_options.copy()

    selected_port_tickers = (
        st.session_state.hedge_portdf["ticker"].dropna().tolist()
    )
    
    for ticker in selected_port_tickers:
        if ticker not in options_pool_y:
            options_pool_y.append(ticker)
    
    
    # X-tickers
    options_pool_x = dropdown_options.copy()

    # Keep all currently selected X tickers available
    for ticker in st.session_state.hedge_assets:
        if ticker not in options_pool_x:
            options_pool_x.append(ticker)


    with st.container(border=True):
        
        st.write("Portfolio holdings:")

        add_col1, add_col2, add_col3 = st.columns([3, 2, 1], width='stretch')
        with add_col1:
            staged_ticker = st.selectbox(
                "Select ticker to add",
                options=dropdown_options,
                key="hedge_staged_ticker",
                label_visibility="collapsed",
                placeholder="pick from search results",
            )
        with add_col2:
            staged_weight = st.number_input(
                label="Weight in %", min_value=0.01, max_value=100.0, step=0.1,
                key="hedge_staged_weight", label_visibility="collapsed",
            )
        with add_col3:
            add_clicked = st.button("Add", key="hedge_add_btn", width='stretch')


        

        if add_clicked:
            current_total = st.session_state.hedge_portdf["weight"].sum()
            if not staged_ticker:
                st.warning("Select a ticker to add.")
            elif staged_ticker in st.session_state.hedge_portdf["ticker"].values:
                st.warning(f"{staged_ticker} is already in the portfolio.")
            elif current_total + staged_weight > 100 + 1e-6:
                st.error(
                    f"Adding {staged_weight:.3f}% would bring the total to "
                    f"{current_total + staged_weight:.3f}%, over the 100% limit. "
                    f"You have {100 - current_total:.3f}% remaining."
                )
            else:
                new_row = pd.DataFrame({"ticker": [staged_ticker], "weight": [staged_weight]})
                st.session_state.hedge_portdf = pd.concat(
                    [st.session_state.hedge_portdf, new_row], ignore_index=True
                )



        st.space("xsmall")


        # options = full static ticker_list, NOT the live search results —
        # this is what stops the column_config from changing shape every rerun
        edited_df = st.data_editor(
            data=st.session_state.hedge_portdf.copy(),
            hide_index=True,
            num_rows="fixed",
            column_config={
                "ticker": st.column_config.SelectboxColumn("Ticker", required=True, options=options_pool_y),
                "weight": st.column_config.NumberColumn(
                    "Weight (%)", min_value=0.0, max_value=100.0, step=0.001, format="%.3f"
                ),
            },
            key="hedge_port_editor",
            width="stretch",
        )


        st.session_state.hedge_portdf = edited_df.copy()

        st.markdown("")

        remove_col1, remove_col2 = st.columns([5,1], width='stretch')
        with remove_col1:
            ticker_to_remove = st.selectbox(
                "Remove a ticker",
                options=st.session_state.hedge_portdf["ticker"].dropna().tolist(),
                key="hedge_remove_ticker",
                label_visibility="collapsed",
                placeholder="select a ticker to remove",
            )
        with remove_col2:
            if st.button("Remove", key="hedge_remove_btn", width='stretch'):
                if ticker_to_remove:
                    st.session_state.hedge_portdf = st.session_state.hedge_portdf[
                        st.session_state.hedge_portdf["ticker"] != ticker_to_remove
                    ].reset_index(drop=True)
                    st.rerun()
    
    
    with st.container(border=True):
        st.write("Hedge instrument ticker(s):")
        assets = st.multiselect(
            label="",
            label_visibility='collapsed',
            options=options_pool_x,
            key='hedge_assets',
            placeholder="select one or many tickers from search"
        )
    
    
    
    # Rest of form
    with st.form("hedge_form", border=False, enter_to_submit=False):
        st.space('xsmall')
        col1, col2 = st.columns(2)

        with col1:
            frequency = st.radio(
                "Choose frequency of data",
                options=["daily", "weekly", "monthly"], key="hedge_frequency", index=0, horizontal=True
            )
        

        with col2:
            period = st.selectbox("Backtest period (optional)", options=['1m','3m','6m','1y','2y','3y','5y','10y','20y','30y'], index=None, key='hedge_period',width='stretch')
        
            
        st.space('xsmall')
        st.markdown("**Custom date range**")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start date for backtest (optional)", value=None, key="hedge_start_date")
        with col2:
            end_date = st.date_input("End date for backtest (optional)", value=None, key="hedge_end_date")
        
        st.space("xsmall")
        st.markdown("### Hedge options")
        col1, col2, col3 = st.columns(3)
        with col1:
            hedge_type = st.radio(
                "Choose hedge type",
                options=["static", "rolling"],
                key="hedge_type",
                index=0,
                horizontal=True,
                persist_state='session'
            )
        with col2:
            static_window = st.number_input(
                "Lookback window for static regression (optional)",
                min_value=12,
                max_value=200,
                value=None,
                step=1,
                key='hedge_static_window'
            )
        with col3:
            rolling_window = st.number_input(
                "Lookback window for rolling regression (optional)",
                min_value=12,
                max_value=125,
                value=None,
                step=1,
                key='hedge_rolling_window'
            )
        
        
        rebalance_freq = st.number_input(
            "How many data observations between each rebalance for rolling hedge (optional)",
            min_value=1,
            max_value=200,
            value=None,
            step=1,
            key='hedge_rebalance_freq'
        )
        
        
        st.space('small')
        
        
        submitted = st.form_submit_button(
            "Run Hedge",
            type="primary",
            use_container_width=True,
        )



if submitted:
    
    port_df = edited_df.copy()
    
    # Basic validation
    if len(port_df) == 0:
        st.error("Please enter at least 1 ticker in portfolio")
    
    
    if len(port_df) > 1:
        port_dict = dict(zip(port_df['ticker'], port_df['weight']))
    elif len(port_df) == 1:
        port_dict = port_df['ticker'][0]
    
    
    if not assets:
        st.error("Please enter hedge assets ticker field.")
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
        
        if hedge_type == 'rolling':
            if isinstance(rolling_window, int):
                if rolling_window < WINDOW_FLOORS[frequency]:
                    rolling_window = WINDOW_DEFAULTS[frequency]
                    st.toast(f"Chosen lookback window for rolling hedge is too low which will reduce hedge effectiveness. Reverted to default {WINDOW_DEFAULTS[period]}.")
            elif rolling_window is None:
                rolling_window = WINDOW_DEFAULTS[frequency]
        elif hedge_type == 'static':
            if isinstance(static_window, int):
                if static_window < WINDOW_FLOORS[frequency]:
                    static_window = WINDOW_DEFAULTS[frequency]
                    st.toast(f"Chosen lookback window for rolling hedge is too low which will reduce hedge effectiveness. Reverted to default {WINDOW_DEFAULTS[period]}.")
            elif static_window is None:
                static_window = WINDOW_DEFAULTS[frequency]
    

        if len(assets) == 1:
            assets = assets[0]
        
        
        try:
            with st.spinner("Fetching data and running hedge sim..."):
                
                if hedge_type == 'static':
                    
                    beta_obj = PortfolioHedge(
                        target=port_dict,
                        hedge_instruments=assets,
                        backtest_period=period,
                        frequency=frequency,
                        backtest_start_date=start_date_str,
                        backtest_end_date=end_date_str,
                        hedge_type="static",
                        static_lookback_window=static_window,
                    )
                    
                elif hedge_type == 'rolling':
                    
                    beta_obj = PortfolioHedge(
                        target=port_dict,
                        hedge_instruments=assets,
                        backtest_period=period,
                        frequency=frequency,
                        backtest_start_date=start_date_str,
                        backtest_end_date=end_date_str,
                        hedge_type="rolling",
                        window=rolling_window,
                        rebalance_freq=rebalance_freq
                    )
                
                beta_obj.backtest(plot=False)
                
                unhedged_port_return_series = beta_obj.unhedged_port_return_series.copy()
                
                if hedge_type == 'rolling':
                    hedged_port_return_series = beta_obj.rolling_hedged_return_series.copy()

                elif hedge_type == 'static':
                    hedged_port_return_series = beta_obj.static_hedged_return_series.copy()
                
                
                metrics_df = beta_obj.metrics_df.copy()
                hedge_df = beta_obj.hedge_df.copy()
                
                
                if hedge_type == 'rolling':
                    rolling_window_final = beta_obj.rolling_window
                    rebalance_freq_final = beta_obj.rebalance_freq
                elif hedge_type == 'static':
                    static_window_final = beta_obj.static_lookback_window
                
                backtest_start = beta_obj.effective_backtest_start_date
                backtest_end = beta_obj.effective_backtest_end_date
                
                
                # Stash everything the display section needs
                st.session_state["hedge_results"] = {
                    "metrics_df": metrics_df,
                    "hedge_df": hedge_df,
                    "unhedged_port_returns_df": unhedged_port_return_series,
                    "hedged_port_returns_df": hedged_port_return_series,
                    "assets": assets,
                    "hedge_type": hedge_type,
                    'frequency': frequency,
                    'backtest_start': backtest_start,
                    'backtest_end': backtest_end,
                    
                }
                
                if hedge_type == 'rolling':
                    st.session_state["hedge_results"]["rolling_window"] = rolling_window_final
                    st.session_state["hedge_results"]["rebalance_freq"] = rebalance_freq_final
                elif hedge_type == 'static':
                    st.session_state["hedge_results"]["static_window"] = static_window_final
                
                
                
        except Exception as e:
            st.error(f"Regression failed: {e}")
            st.session_state.pop("results", None)
            time.sleep(5)
            st.rerun()



if 'hedge_results' in st.session_state:
    r = st.session_state["hedge_results"]
    metrics_df = r['metrics_df']
    hedge_df = r['hedge_df']
    unhedged_returns = r['unhedged_port_returns_df']
    hedged_returns = r['hedged_port_returns_df']
    assets = r['assets']
    hedge_type = r['hedge_type']
    freq = r['frequency']
    start = r['backtest_start']
    end = r['backtest_end']
    
    if hedge_type == 'rolling':
        rolling_window = r['rolling_window']
        rebalance_freq = r['rebalance_freq']
    elif hedge_type == 'static':
        static_window = r['static_window']
    
    
    st.space('small')
    
    st.subheader('Hedge Results')
    
    if hedge_type == 'static':
        st.markdown(f"{hedge_type.title()} hedge with a lookback window of {static_window} {freq} observations. Backtest is from {start} to {end}.")
    elif hedge_type == 'rolling':
        st.markdown(f"{hedge_type.title()} hedge with a lookback window of {rolling_window} {freq} observations and a rebalance frequency of every {rebalance_freq} observations. Backtest is from {start} to {end}.")
    
    st.markdown(f"The portfolio is hedged by shorting {", ".join(ticker for ticker in assets)}.")
    
    
    # Portfolio cumulative returns plot
    st.space("small")
    st.markdown('##### Hedged vs Unhedged Cumulative Portfolio Returns')
    cum_df = hedged_returns.copy().set_index("date")[['port-returns','hedged-returns']]
    cum_df = (1 + cum_df).cumprod() - 1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df['port-returns'], mode="lines", name=f'Portfolio unhedged returns',
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df['hedged-returns'], mode="lines", name=f'Portfolio hedged returns',
        line=dict(width=2),
    ))
    
    fig.update_layout(
        #title="Cumulative Returns",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, l=10, r=10, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch')



    # Drawdown plot
    cum_unhedged_df = hedged_returns.copy().set_index("date")[['port-returns']]
    cum_hedged_df = hedged_returns.copy().set_index("date")[['hedged-returns']]
    
    unhedged_wealth = (1 + cum_unhedged_df).cumprod()
    hedged_wealth = (1 + cum_hedged_df).cumprod()
    
    running_max_unhedged = unhedged_wealth.cummax()
    running_max_hedged = hedged_wealth.cummax()
    
    unhedged_drawdowns = unhedged_wealth / running_max_unhedged - 1
    hedged_drawdowns = hedged_wealth / running_max_hedged - 1
    
    
    st.space("small")
    st.markdown('##### Portfolio Drawdowns')
    
    
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=unhedged_drawdowns.index, y=unhedged_drawdowns['port-returns'], mode="lines", name=f'Unhedged Portfolio Drawdowns',
        line=dict(width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=hedged_drawdowns.index, y=hedged_drawdowns['hedged-returns'], mode="lines", name=f'Hedged Portfolio Drawdowns',
        line=dict(width=2),
    ))
    fig1.update_layout(
        #title="Portfolio Drawdowns",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, l=10, r=10, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig1, width='stretch')
    
    
    
    
    
    
    # Metrics

    
    h = 'Hedged'
    u = 'Unhedged'
    
    max_draw_h = float(metrics_df.loc['Maximum Drawdown',h])
    #max_draw_u = float(metrics_df.loc['Maximum Drawdown',u])
    max_draw_d = float(metrics_df.loc['Maximum Drawdown','Improvement'])
    max_draw_dp = float(hedge_df.loc['Maximum Drawdown Reduction','Value'])
    
    ann_vol_h = float(metrics_df.loc['Annualized Volatility',h])
    #ann_vol_u = float(metrics_df.loc['Annualized Volatility',u])
    ann_vol_d = float(metrics_df.loc['Annualized Volatility','Improvement'])
    ann_vol_dp = float(hedge_df.loc['Volatility Reduction','Value'])
    

    st.space('small')
    st.markdown('##### Drawdown and Volatility')
    
    col1, col2 = st.columns(2)
    
    col1.metric(
        "Max Drawdown",
        f"{max_draw_h:.2%}",
        border=True,
        delta=f"{-max_draw_d:.2%} vs unhedged",
        delta_color="inverse",
        #delta_arrow = 'off'
    )
    
    col2.metric(
        "Annualized Volatility",
        f"{ann_vol_h:.2%}",
        border=True,
        delta=f"{-ann_vol_d:.2%} vs unhedged",
        delta_color="inverse",
        #delta_arrow = 'off'
    
    )
    
    col1, col2 = st.columns(2)
    
    col1.metric(
        "Max Drawdown Reduction",
        f"{max_draw_dp:.2%}",
        border=True,
    )
    
    col2.metric(
        "Annualized Volatility Reduction",
        f"{ann_vol_dp:.2%}",
        border=True,
    )
    
    
    sharpe_h = float(metrics_df.loc['Sharpe Ratio',h])
    sharpe_d = float(metrics_df.loc['Sharpe Ratio','Improvement'])
    
    sortino_h = float(metrics_df.loc['Sortino Ratio',h])
    sortino_d = float(metrics_df.loc['Sortino Ratio','Improvement'])
    
    calmar_h = float(metrics_df.loc['Calmar Ratio',h])
    calmar_d = float(metrics_df.loc['Calmar Ratio','Improvement'])
    
    
    st.space('xxsmall')
    st.markdown('##### Ratios')
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Sharpe Ratio",
        f"{sharpe_h:.3f}",
        border=True,
        delta = f"{sharpe_d:.3f} vs unhedged",
    )
    
    col2.metric(
        "Sortino Ratio",
        f"{sortino_h:.3f}",
        border=True,
        delta = f"{sortino_d:.3f} vs unhedged",
    )
    
    col3.metric(
        "Calmar Ratio",
        f"{calmar_h:.3f}",
        border=True,
        delta = f"{calmar_d:.3f} vs unhedged",
    )
    
    st.space('xxsmall')
    st.markdown('##### Returns')
    col1, col2 = st.columns(2)
    
    tr_h = float(metrics_df.loc['Total Return',h])
    tr_d = float(metrics_df.loc['Total Return','Improvement'])
    
    ar_h = float(metrics_df.loc['Annualized Return',h])
    ar_d = float(metrics_df.loc['Annualized Return','Improvement'])
    
    col1.metric(
        "Total Return",
        f"{tr_h:.2%}",
        border=True,
        delta = f"{tr_d:.2%} vs unhedged",
    )
    
    col2.metric(
        "Annualized Return",
        f"{ar_h:.2%}",
        border=True,
        delta = f"{ar_d:.2%} vs unhedged",
    )
    
    st.space('xxsmall')
    st.markdown('##### Value at Risk Metrics')
    
    
    v_h = float(metrics_df.loc['VaR (95%)',h])
    v_d = float(metrics_df.loc['VaR (95%)','Improvement'])
    v_dp = float(hedge_df.loc['VaR Reduction (95%)','Value'])
    
    cv_h = float(metrics_df.loc['CVaR (95%)',h])
    cv_d = float(metrics_df.loc['CVaR (95%)','Improvement'])
    cv_dp = float(hedge_df.loc['CVaR Reduction (95%)','Value'])
    
    
    col1, col2 = st.columns(2)
    col1.metric(
        "VaR (95%)",
        f"{v_h:.2%}",
        border=True,
        delta = f"{v_d:.2%} vs unhedged",
        #delta_arrow = 'off'
    )
    
    col2.metric(
        "CVaR (95%)",
        f"{cv_h:.2%}",
        border=True,
        delta = f"{cv_d:.2%} vs unhedged",
        #delta_arrow = 'off'
    )
    
    col1, col2 = st.columns(2)
    col1.metric(
        "VaR (95%) reduction",
        f"{v_dp:.2%}",
        border=True,
    )
    
    col2.metric(
        "CVaR (95%) reduction",
        f"{cv_dp:.2%}",
        border=True,
    )
    
    
    #col2.metric("N Obs", f"{int(metrics_df["n_obs"])}", border=True)
    
    
    # Raw data dataframes
    st.space('medium')
    st.markdown('##### Raw Data Tables')
    with st.expander(label='Raw Data'):
        
        st.markdown('##### Risk Metrics Results')
        
        st.dataframe(metrics_df)
        with st.container(horizontal=True, horizontal_alignment='right'):
            st.download_button("Download CSV", metrics_df.to_csv(index=True), f"hedge_metrics.csv", key='metrics_download_csv')

            rol_js_str = metrics_df.reset_index().to_json(orient="records", indent=4, date_format='iso')

        
            st.download_button(
                label="Download JSON",
                data=rol_js_str,
                file_name=f"hedge_metrics.json",
                mime="application/json",
                key='metrics_download_json'
            )
        
        
        
        st.space('small')
        st.markdown('##### Risk Metrics Changes')
        st.dataframe(hedge_df)
        with st.container(horizontal=True, horizontal_alignment='right'):
            st.download_button("Download CSV", hedge_df.to_csv(index=True), f"hedge_improvement.csv", key='hedge_download_csv')

            rol_js_str = hedge_df.reset_index().to_json(orient="records", indent=4, date_format='iso')

        
            st.download_button(
                label="Download JSON",
                data=rol_js_str,
                file_name=f"hedge_improvement.json",
                mime="application/json",
                key='hedge_download_json'
            )
        
        
        
        st.space('small')
        st.markdown('##### Underlying Returns Data')
        st.dataframe(hedged_returns, hide_index=True)
        with st.container(horizontal=True, horizontal_alignment='right'):
            st.download_button("Download CSV", hedged_returns.to_csv(index=False), f"returns_data.csv", key='returns_download_csv')

            rol_js_str = hedged_returns.to_json(orient="records", indent=4, date_format='iso')

        
            st.download_button(
                label="Download JSON",
                data=rol_js_str,
                file_name=f"returns_data.json",
                mime="application/json",
                key='returns_download_json'
            )
                
        
    st.space('medium')
    
    # Reset button
    if st.button("Reset Regression"):
        keys_to_clear = [
            "hedge_portdf", "hedge_assets", "hedge_frequency", "hedge_period",
            "hedge_start_date", "hedge_end_date", "hedge_results",'hedge_search_box','hedge_staged_ticker',
            "hedge_staged_weight", "hedge_add_btn","hedge_port_editor", "hedge_remove_ticker","hedge_remove_btn",
            'hedge_assets',"hedge_form", "hedge_type", "hedge_static_window", "hedge_rolling_window", "hedge_rebalance_freq"
            ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)

        
        st.rerun()
        




