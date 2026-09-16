import streamlit as st
import pandas as pd
import numpy as np
import datetime
from beta_tool.regression_beta.beta import Beta
import plotly.express as px
import plotly.graph_objects as go
from beta_tool.webapp import theme
#from beta_tool.webapp.plots import rolling_beta_chart




st.set_page_config(
    page_title="Single Asset Beta",
    page_icon="📈",
    layout="wide",
)

st.title("Single Asset Beta")

st.markdown(
    """
    Calculate the beta of one asset's returns against another
    using ordinary least squares (OLS) regression.
    
    You can also choose the return frequency, return methodology,
    observation period, and HAC-aware standard errors.
    """
)

st.markdown("")




with st.form("beta_form", border=True, enter_to_submit=False):
    st.subheader("Regression inputs")

    col1, col2 = st.columns(2)
    with col1:
        asset1 = st.text_input("Dependent (y) ticker:", placeholder="aapl", key="asset1")
    with col2:
        asset2 = st.text_input("Independent (x) ticker:", placeholder="spy", key="asset2")


    col1, col2 = st.columns(2)

    with col1:
        frequency = st.radio(
            "Choose frequency of data",
            options=["daily", "weekly", "monthly"], key="frequency", index=0, horizontal=True
        )
    with col2:
        return_type = st.radio(
            "Choose how returns are calculated",
            options=["log", "simple"], key="return_type", index=1, horizontal=True
        )


    col1, col2 = st.columns(2)

    with col1:
        period = st.selectbox("Regression period", options=['1m','3m','6m','1y','2y','3y','5y','10y','20y','30y'], index=3, key='period')
    
        

    st.markdown("**Custom date range**")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start date for regression (Optional)", value=None, key="start_date")
    with col2:
        end_date = st.date_input("End date for regression (Optional)", value=None, key="end_date")

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
            key='hac'
        )

    with col2:
        hac_lag = st.number_input(
            "HAC lags",
            min_value=1,
            max_value=40,
            value=5,
            step=1,
            key='hac_lag'
        )

    st.markdown("")
    st.markdown("#### Rolling Beta")

    col1, col2 = st.columns(2)

    with col1:
        rolling = st.radio(
            "Do rolling beta?",
            options=[True,False],
            format_func= lambda x: "Yes" if x else "No",
            horizontal=True,
            index=0,
            key="rolling"
        )

    with col2:
        rolling_window = st.number_input(
            "Lookback window for rolling Beta",
            min_value=2,
            max_value=200,
            value=60,
            step=1,
            key="rolling_window"
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
    if not asset1.strip() or not asset2.strip():
        st.error("Please enter both asset tickers.")

    elif asset1.strip().upper() == asset2.strip().upper():
        st.error("The dependent and independent assets must be different.")

    elif (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        st.error("Start date must be earlier than end date.")

    else:

        # Convert tickers to lowercase
        asset1 = asset1.strip().lower()
        asset2 = asset2.strip().lower()

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

        with st.spinner("Fetching data and running regression..."):

            try:
                beta_obj = Beta(
                    asset1=asset1,
                    asset2=asset2,
                    period=period,
                    frequency=frequency,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    return_type=return_type,
                    hac=hac,
                    hac_lag=selected_hac_lag,
                )

                # Results dataframe
                df = beta_obj.ols_df.copy()

                # Returns dataframes
                
                y_col = beta_obj.y_col
                x_col = beta_obj.x_col
                returns_df = beta_obj.returns_df.copy()

                returns_df = returns_df.rename(columns={y_col: f"{asset1.upper()}-returns", x_col: f"{asset2.upper()}-returns"})

                y_col = f"{asset1.upper()}-returns"
                x_col = f"{asset2.upper()}-returns"


                

                # Stash everything the display section needs
                st.session_state["results"] = {
                    "df": df,
                    "returns_df": returns_df,
                    "y_col": y_col,
                    "x_col": x_col,
                    "asset1": asset1,
                    "asset2": asset2,
                }

                if rolling:
                    beta_obj.historical_rolling_beta(window=rolling_window)
                    rolling_df = beta_obj.rolling_df.copy()
                    st.session_state["results"]["rolling_df"] = rolling_df

                

            except Exception as e:
                st.error(f"Regression failed: {e}")
                st.session_state.pop("results", None)
                st.stop()


if 'results' in st.session_state:
    r = st.session_state["results"]
    df, returns_df = r["df"], r["returns_df"]
    x_col, y_col = r["x_col"], r["y_col"]
    asset1, asset2 = r["asset1"], r["asset2"]

    if rolling:
        r_df = r["rolling_df"]

    st.subheader('Results')

    # One line summary
    beta_val = float(df['beta'].iloc[0])
    pval = float(df['beta_pvalue'].iloc[0])
    sig = "statistically significant" if pval < 0.05 else "not statistically significant at the 5% level"
    st.caption(
        f"A 1% move in {asset2.upper()} is associated with a {beta_val:.2f}% move in {asset1.upper()}, "
        f"on average ({sig})."
    )

    st.write()

    # Raw results metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beta", f"{beta_val:.3f}", delta=f"{beta_val - 1:.3f} vs 1.0", delta_color="off", border=True)
    col2.metric("R²", f"{float(df['r_squared'].iloc[0]):.3f}", border=True, height='stretch')
    col3.metric("P-value", f"{float(df["beta_pvalue"].iloc[0]):.2e}", border=True, height='stretch')
    col4.metric("N Obs", f"{int(df["n_obs"].iloc[0])}", border=True, height='stretch')

    with st.expander("Full regression stats"):
        st.dataframe(df, width='stretch')

        col1, col2, _ = st.columns([1,1,3])
        with col1:
            st.download_button("Download CSV", df.to_csv(index=False), "regression_stats.csv")

        df_js_str = df.to_json(orient="records", indent=4,date_format='iso')

        with col2:
            st.download_button(
                label="Download JSON",
                data=df_js_str,
                file_name="regression_stats.json",
                mime="application/json"
            )

    # Returns visualisation
    fig = px.scatter(
        returns_df, x=x_col, y=y_col,
        trendline="ols", opacity=0.5,
        labels={"asset_2_returns": asset2.upper(), "asset_1_returns": asset1.upper()},
        title=f"{asset1.upper()} vs {asset2.upper()} — {frequency} returns",
    )

    fig.update_traces(line=dict(color="#D9663A", width=2), selector=dict(mode="lines"))
    fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
    
    fig.update_layout(
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        title_font_size=18,
        margin=dict(t=60, l=10, r=10, b=10),
    )

    st.plotly_chart(fig, width='stretch')

    # Cumulative returns plot
    #st.subheader("Cumulative returns")
    cum_df = returns_df.set_index("date")[[x_col, y_col]].rename(
        columns={x_col: asset2.upper(), y_col: asset1.upper()}
    )
    if return_type == "log":
        cum_df = np.exp(cum_df.cumsum()) - 1
    else:
        cum_df = (1 + cum_df).cumprod() - 1

    fig2 = go.Figure()
    for col, color in zip(cum_df.columns, ["#4C78A8", "#E45756"]):
        fig2.add_trace(go.Scatter(
            x=cum_df.index, y=cum_df[col], mode="lines", name=col,
            line=dict(width=2),
        ))
    fig2.update_layout(
        title="Cumulative Returns",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, l=10, r=10, b=10),
        hovermode="x unified",
    )

    st.plotly_chart(fig2, width='stretch')

    # st.line_chart(cum_df)

    # Raw returns series download
    with st.expander("Raw returns data"):
        st.dataframe(returns_df, width='stretch')

        col1, col2, _ = st.columns([1,1,3])
        with col1:
            st.download_button("Download CSV", returns_df.to_csv(index=False), "returns.csv")
                
        returns_js_str = returns_df.to_json(orient="records", indent=4, date_format='iso')

        with col2:
            st.download_button(
                label="Download JSON",
                data=returns_js_str,
                file_name="returns.json",
                mime="application/json"
            )

        

    
    # Rolling beta data
    if rolling:
        st.markdown("")
        st.markdown("##### Rolling Beta Results")

        st.markdown("")

        # Rolling beta plot
        rol_df = r_df.set_index("date")[["beta","beta_ci_upper","beta_ci_lower"]]
        fig3 = go.Figure()

        

        # Confidence interval
        # 1. Add the lower bound trace (hidden line, used as the fill anchor)
        fig3.add_trace(go.Scatter(
            x=rol_df.index,
            y=rol_df['beta_ci_lower'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            #hoverinfo='skip',
            name='95% CI lower',
            fillcolor='rgba(0, 0, 0, 0.1)',
        ))

        # 2. Add the upper bound trace and fill down to the lower bound trace
        fig3.add_trace(go.Scatter(
            x=rol_df.index,
            y=rol_df['beta_ci_upper'],
            mode='lines',
            line= dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 0, 0, 0.1)',  # Semi-transparent color for the band
            name='95% CI upper',
            showlegend=False,
        ))

        # Beta time series
        fig3.add_trace(go.Scatter(
            x=rol_df.index, y=rol_df["beta"], mode="lines", name="beta",
            line=dict(width=2),
        ))


        fig3.update_layout(
            title=f"{rolling_window}-observations Rolling Beta with 95% Confidence Intervals",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, l=10, r=10, b=10),
            hovermode="x unified",
        )

        

        st.plotly_chart(fig3, width='stretch')

        # Rolling Beta Dataframe
        st.markdown("")
        with st.expander("Rolling Beta Stats"):
            st.dataframe(r_df, width='stretch')

            col1, col2, _ = st.columns([1,1,3])

            with col1:
                st.download_button("Download CSV", r_df.to_csv(index=False), "rolling_stats.csv")

            rol_js_str = r_df.to_json(orient="records", indent=4, date_format='iso')

            with col2:
                st.download_button(
                    label="Download JSON",
                    data=rol_js_str,
                    file_name=f"{asset1}_{asset2}_{rolling_window}_rolling_beta_stats.json",
                    mime="application/json"
                )

        st.markdown("")
        
    st.markdown("")
    st.markdown("")

    # Reset button
    if st.button("Reset Regression", width='stretch'):
        keys_to_clear = [
            "asset1", "asset2", "frequency", "return_type", "period",
            "start_date", "end_date", "hac", "hac_lag", "results",
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.rerun()




