import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Portfolio Risk & Hedge Analytics
# =============================================================================

def calculate_risk_metrics(
    df,
    return_col="port-returns",
    hedged_col="hedged-returns",
    date_col="date",
    frequency="daily",
    risk_free_rate=0.0,
    var_confidence=0.95,
    trading_days=252,
    print_report=True,
    plot=False,
):
    """
    Calculate comprehensive risk and hedge-effectiveness metrics for an
    unhedged vs. hedged portfolio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing portfolio returns.

    return_col : str, default="port-returns"
        Column containing unhedged portfolio returns.

    hedged_col : str, default="hedged-returns"
        Column containing hedged portfolio returns.

    date_col : str, default="date"
        Date column.

    frequency : {"daily", "weekly", "monthly"}
        Frequency of the return series.

    risk_free_rate : float, default=0.0
        Annualized risk-free rate.
        Example: 0.04 = 4%.

    var_confidence : float, default=0.95
        Confidence level for VaR/CVaR.

    trading_days : int, default=252
        Number of trading days per year.
        Used for daily data.

    print_report : bool, default=True
        If True, print a formatted report.

    plot : bool, default=False
        If True, display cumulative-return and drawdown charts.

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame containing unhedged, hedged, change, and improvement metrics.

    summary : dict
        Additional hedge-effectiveness statistics.

    """

    # =========================================================================
    # Validation
    # =========================================================================

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = [return_col, hedged_col]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    frequency = frequency.lower()

    periods_per_year = {
        "daily": trading_days,
        "weekly": 52,
        "monthly": 12,
    }

    if frequency not in periods_per_year:
        raise ValueError(
            "frequency must be one of: "
            "'daily', 'weekly', or 'monthly'."
        )

    annualization_factor = periods_per_year[frequency]

    if not 0 < var_confidence < 1:
        raise ValueError("var_confidence must be between 0 and 1.")

    # =========================================================================
    # Prepare data
    # =========================================================================

    data = df.copy()

    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(date_col)

    data[return_col] = pd.to_numeric(
        data[return_col],
        errors="coerce"
    )

    data[hedged_col] = pd.to_numeric(
        data[hedged_col],
        errors="coerce"
    )

    data = data[[return_col, hedged_col]].dropna()

    if len(data) < 2:
        raise ValueError(
            "At least two valid observations are required."
        )

    unhedged = data[return_col]
    hedged = data[hedged_col]

    # Periodic risk-free rate
    periodic_rf = (
        (1 + risk_free_rate) ** (1 / annualization_factor)
    ) - 1

    # =========================================================================
    # Helper functions
    # =========================================================================

    def total_return(returns):
        return (1 + returns).prod() - 1

    def annualized_return(returns):
        n = len(returns)

        if n == 0:
            return np.nan

        total = total_return(returns)

        # Avoid invalid values when cumulative wealth <= 0
        if 1 + total <= 0:
            return np.nan

        return (1 + total) ** (
            annualization_factor / n
        ) - 1

    def annualized_volatility(returns):
        return returns.std(ddof=1) * np.sqrt(
            annualization_factor
        )

    def downside_deviation(returns):
        downside = returns[returns < periodic_rf]

        if len(downside) == 0:
            return 0.0

        return np.sqrt(
            np.mean((downside - periodic_rf) ** 2)
        ) * np.sqrt(annualization_factor)

    def sharpe_ratio(returns):
        volatility = returns.std(ddof=1)

        if volatility == 0:
            return np.nan

        return (
            (returns.mean() - periodic_rf)
            / volatility
            * np.sqrt(annualization_factor)
        )

    def sortino_ratio(returns):
        downside = returns[returns < periodic_rf]

        if len(downside) == 0:
            return np.nan

        downside_risk = np.sqrt(
            np.mean((downside - periodic_rf) ** 2)
        )

        if downside_risk == 0:
            return np.nan

        return (
            (returns.mean() - periodic_rf)
            / downside_risk
            * np.sqrt(annualization_factor)
        )

    def drawdown_statistics(returns):

        wealth = (1 + returns).cumprod()

        running_max = wealth.cummax()

        drawdowns = wealth / running_max - 1

        max_drawdown = drawdowns.min()

        # ---------------------------------------------------------------------
        # Drawdown duration
        # ---------------------------------------------------------------------

        duration = 0
        max_duration = 0

        for dd in drawdowns:

            if dd < 0:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0

        # ---------------------------------------------------------------------
        # Average drawdown
        # ---------------------------------------------------------------------

        negative_drawdowns = drawdowns[drawdowns < 0]

        average_drawdown = (
            negative_drawdowns.mean()
            if len(negative_drawdowns) > 0
            else 0.0
        )

        return (
            max_drawdown,
            max_duration,
            average_drawdown,
            drawdowns,
            wealth,
        )

    def calmar_ratio(returns):

        ann_return = annualized_return(returns)

        max_dd = drawdown_statistics(returns)[0]

        if max_dd == 0:
            return np.nan

        return ann_return / abs(max_dd)

    def var(returns):
        """
        Historical VaR.

        Negative number represents a loss.
        """

        return returns.quantile(1 - var_confidence)

    def cvar(returns):

        value_at_risk = var(returns)

        tail = returns[returns <= value_at_risk]

        if len(tail) == 0:
            return np.nan

        return tail.mean()

    def beta(y, x):

        covariance = np.cov(y, x, ddof=1)[0, 1]
        variance = np.var(x, ddof=1)

        if variance == 0:
            return np.nan

        return covariance / variance

    def tracking_error(hedged_returns, benchmark_returns):

        active_returns = (
            hedged_returns - benchmark_returns
        )

        return active_returns.std(ddof=1) * np.sqrt(
            annualization_factor
        )

    def information_ratio(hedged_returns, benchmark_returns):

        active_returns = (
            hedged_returns - benchmark_returns
        )

        te = active_returns.std(ddof=1)

        if te == 0:
            return np.nan

        return (
            active_returns.mean()
            / te
            * np.sqrt(annualization_factor)
        )

    def positive_period_rate(returns):
        return (returns > 0).mean()

    def negative_period_rate(returns):
        return (returns < 0).mean()

    def average_gain(returns):

        gains = returns[returns > 0]

        if len(gains) == 0:
            return np.nan

        return gains.mean()

    def average_loss(returns):

        losses = returns[returns < 0]

        if len(losses) == 0:
            return np.nan

        return losses.mean()

    def gain_loss_ratio(returns):

        gain = average_gain(returns)
        loss = average_loss(returns)

        if pd.isna(gain) or pd.isna(loss) or loss == 0:
            return np.nan

        return abs(gain / loss)

    # =========================================================================
    # Calculate metrics for each portfolio
    # =========================================================================

    def calculate_metrics(returns):

        (
            max_dd,
            max_dd_duration,
            average_dd,
            drawdowns,
            wealth,
        ) = drawdown_statistics(returns)

        ann_return = annualized_return(returns)

        metrics = {

            # -----------------------------------------------------------------
            # Return
            # -----------------------------------------------------------------

            "Observations": len(returns),

            "Total Return": total_return(returns),

            "Annualized Return": ann_return,

            # -----------------------------------------------------------------
            # Risk
            # -----------------------------------------------------------------

            "Annualized Volatility":
                annualized_volatility(returns),

            "Downside Deviation":
                downside_deviation(returns),

            "Maximum Drawdown":
                max_dd,

            "Average Drawdown":
                average_dd,

            "Max Drawdown Duration":
                max_dd_duration,

            # -----------------------------------------------------------------
            # Risk-adjusted return
            # -----------------------------------------------------------------

            "Sharpe Ratio":
                sharpe_ratio(returns),

            "Sortino Ratio":
                sortino_ratio(returns),

            "Calmar Ratio":
                calmar_ratio(returns),

            # -----------------------------------------------------------------
            # Tail risk
            # -----------------------------------------------------------------

            f"VaR ({var_confidence:.0%})":
                var(returns),

            f"CVaR ({var_confidence:.0%})":
                cvar(returns),

            "Worst Period":
                returns.min(),

            "Best Period":
                returns.max(),

            # -----------------------------------------------------------------
            # Distribution
            # -----------------------------------------------------------------

            "Skewness":
                returns.skew(),

            "Kurtosis":
                returns.kurtosis(),

            # -----------------------------------------------------------------
            # Hit rate / win-loss
            # -----------------------------------------------------------------

            "Positive Period Rate":
                positive_period_rate(returns),

            "Negative Period Rate":
                negative_period_rate(returns),

            "Average Gain":
                average_gain(returns),

            "Average Loss":
                average_loss(returns),

            "Gain/Loss Ratio":
                gain_loss_ratio(returns),
        }

        return metrics, drawdowns, wealth

    # =========================================================================
    # Calculate
    # =========================================================================

    unhedged_metrics, unhedged_dd, unhedged_wealth = (
        calculate_metrics(unhedged)
    )

    hedged_metrics, hedged_dd, hedged_wealth = (
        calculate_metrics(hedged)
    )

    # =========================================================================
    # Main metrics DataFrame
    # =========================================================================

    metrics_df = pd.DataFrame(
        {
            "Unhedged": unhedged_metrics,
            "Hedged": hedged_metrics,
        }
    )

    metrics_df["Change"] = (
        metrics_df["Hedged"]
        - metrics_df["Unhedged"]
    )

    # =========================================================================
    # Hedge effectiveness
    # =========================================================================

    volatility_reduction = (
        1
        - (
            hedged_metrics["Annualized Volatility"]
            / unhedged_metrics["Annualized Volatility"]
        )
    )

    downside_risk_reduction = (
        1
        - (
            hedged_metrics["Downside Deviation"]
            / unhedged_metrics["Downside Deviation"]
        )
        if unhedged_metrics["Downside Deviation"] != 0
        else np.nan
    )

    # Drawdown reduction:
    #
    # Example:
    # Unhedged = -30%
    # Hedged   = -20%
    #
    # Reduction = 1 - 20/30 = 33.3%

    if unhedged_metrics["Maximum Drawdown"] != 0:

        drawdown_reduction = (
            1
            - (
                abs(hedged_metrics["Maximum Drawdown"])
                / abs(unhedged_metrics["Maximum Drawdown"])
            )
        )

    else:
        drawdown_reduction = np.nan

    # Tail-loss reduction

    unhedged_cvar = abs(
        unhedged_metrics[f"CVaR ({var_confidence:.0%})"]
    )

    hedged_cvar = abs(
        hedged_metrics[f"CVaR ({var_confidence:.0%})"]
    )

    cvar_reduction = (
        1 - hedged_cvar / unhedged_cvar
        if unhedged_cvar != 0
        else np.nan
    )

    # VaR reduction

    unhedged_var = abs(
        unhedged_metrics[f"VaR ({var_confidence:.0%})"]
    )

    hedged_var = abs(
        hedged_metrics[f"VaR ({var_confidence:.0%})"]
    )

    var_reduction = (
        1 - hedged_var / unhedged_var
        if unhedged_var != 0
        else np.nan
    )

    # Return sacrificed

    return_difference = (
        hedged_metrics["Annualized Return"]
        - unhedged_metrics["Annualized Return"]
    )

    # Volatility reduction per unit of annualized return sacrificed

    if return_difference < 0 and volatility_reduction > 0:

        risk_reduction_per_return = (
            volatility_reduction
            / abs(return_difference)
        )

    else:
        risk_reduction_per_return = np.nan

    # Correlation

    return_correlation = unhedged.corr(hedged)

    # Beta of hedged portfolio relative to unhedged

    hedge_beta = beta(hedged.values, unhedged.values)

    # Tracking error

    te = tracking_error(hedged, unhedged)

    # Information ratio

    ir = information_ratio(hedged, unhedged)

    # Return correlation reduction is not really meaningful by itself,
    # but beta tells us how much of the unhedged portfolio movement
    # remains in the hedged portfolio.

    beta_reduction = 1 - hedge_beta if not pd.isna(hedge_beta) else np.nan

    # =========================================================================
    # Add hedge effectiveness to DataFrame
    # =========================================================================

    hedge_metrics = {

        "Volatility Reduction": volatility_reduction,

        "Downside Risk Reduction": downside_risk_reduction,

        "Maximum Drawdown Reduction": drawdown_reduction,

        f"VaR Reduction ({var_confidence:.0%})": var_reduction,

        f"CVaR Reduction ({var_confidence:.0%})": cvar_reduction,

        "Annualized Return Difference": return_difference,

        "Risk Reduction / Return Sacrificed":
            risk_reduction_per_return,

        "Return Correlation":
            return_correlation,

        "Hedge Beta":
            hedge_beta,

        "Beta Reduction":
            beta_reduction,

        "Tracking Error":
            te,

        "Information Ratio":
            ir,
    }

    hedge_df = pd.DataFrame.from_dict(
        hedge_metrics,
        orient="index",
        columns=["Value"]
    )

    # =========================================================================
    # Improvement calculation
    # =========================================================================

    #
    # Positive improvement means the hedge helped.
    #

    higher_is_better = [
        "Total Return",
        "Annualized Return",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Positive Period Rate",
        "Gain/Loss Ratio",
    ]

    lower_is_better = [
        "Annualized Volatility",
        "Downside Deviation",
        "Maximum Drawdown",
        "Average Drawdown",
        "Max Drawdown Duration",
        f"VaR ({var_confidence:.0%})",
        f"CVaR ({var_confidence:.0%})",
        "Worst Period",
        "Negative Period Rate",
    ]

    metrics_df["Improvement"] = np.nan

    for metric in metrics_df.index:

        if metric in higher_is_better:

            metrics_df.loc[metric, "Improvement"] = (
                metrics_df.loc[metric, "Hedged"]
                - metrics_df.loc[metric, "Unhedged"]
            )

        elif metric in lower_is_better:

            # For risk metrics, a reduction is an improvement.
            #
            # Example:
            # volatility 20% -> 15%
            # improvement = +5%

            if metric == "Maximum Drawdown":
                metrics_df.loc[metric, "Improvement"] = (
                    abs(metrics_df.loc[metric, "Unhedged"])
                    - abs(metrics_df.loc[metric, "Hedged"])
                )

            elif metric == "Average Drawdown":
                metrics_df.loc[metric, "Improvement"] = (
                    abs(metrics_df.loc[metric, "Unhedged"])
                    - abs(metrics_df.loc[metric, "Hedged"])
                )

            elif metric in [
                f"VaR ({var_confidence:.0%})",
                f"CVaR ({var_confidence:.0%})",
                "Worst Period",
            ]:
                metrics_df.loc[metric, "Improvement"] = (
                    abs(metrics_df.loc[metric, "Unhedged"])
                    - abs(metrics_df.loc[metric, "Hedged"])
                )

            else:
                metrics_df.loc[metric, "Improvement"] = (
                    metrics_df.loc[metric, "Unhedged"]
                    - metrics_df.loc[metric, "Hedged"]
                )

    # =========================================================================
    # Formatted printing
    # =========================================================================

    def fmt_percent(x):

        if pd.isna(x):
            return "N/A"

        return f"{x:,.2%}"

    def fmt_number(x):

        if pd.isna(x):
            return "N/A"

        return f"{x:,.2f}"

    def print_report():

        width = 78

        print()
        print("=" * width)
        print("             PORTFOLIO RISK & HEDGE ANALYSIS")
        print("=" * width)

        print()
        print(
            f"Frequency       : {frequency.capitalize()}"
        )

        print(
            f"Observations    : {len(data):,}"
        )

        print(
            f"VaR Confidence  : {var_confidence:.0%}"
        )

        print(
            f"Risk-Free Rate  : {risk_free_rate:.2%}"
        )

        print()
        print("-" * width)
        print("RETURN & RISK")
        print("-" * width)

        rows = [
            ("Total Return", "percent"),
            ("Annualized Return", "percent"),
            ("Annualized Volatility", "percent"),
            ("Downside Deviation", "percent"),
            ("Sharpe Ratio", "number"),
            ("Sortino Ratio", "number"),
            ("Calmar Ratio", "number"),
            ("Maximum Drawdown", "percent"),
            ("Average Drawdown", "percent"),
            ("Max Drawdown Duration", "number"),
            (f"VaR ({var_confidence:.0%})", "percent"),
            (f"CVaR ({var_confidence:.0%})", "percent"),
            ("Worst Period", "percent"),
            ("Best Period", "percent"),
        ]

        print(
            f"{'Metric':<30}"
            f"{'Unhedged':>15}"
            f"{'Hedged':>15}"
            f"{'Change':>15}"
        )

        print("-" * width)

        for metric, fmt in rows:

            u = metrics_df.loc[metric, "Unhedged"]
            h = metrics_df.loc[metric, "Hedged"]
            c = metrics_df.loc[metric, "Change"]

            if fmt == "percent":
                u = fmt_percent(u)
                h = fmt_percent(h)
                c = fmt_percent(c)
            else:
                u = fmt_number(u)
                h = fmt_number(h)
                c = fmt_number(c)

            print(
                f"{metric:<30}"
                f"{u:>15}"
                f"{h:>15}"
                f"{c:>15}"
            )

        print()
        print("-" * width)
        print("HEDGE EFFECTIVENESS")
        print("-" * width)

        effectiveness_rows = [
            (
                "Volatility Reduction",
                volatility_reduction,
                "percent",
            ),
            (
                "Downside Risk Reduction",
                downside_risk_reduction,
                "percent",
            ),
            (
                "Maximum Drawdown Reduction",
                drawdown_reduction,
                "percent",
            ),
            (
                f"VaR Reduction ({var_confidence:.0%})",
                var_reduction,
                "percent",
            ),
            (
                f"CVaR Reduction ({var_confidence:.0%})",
                cvar_reduction,
                "percent",
            ),
            (
                "Annualized Return Difference",
                return_difference,
                "percent",
            ),
            (
                "Return Correlation",
                return_correlation,
                "number",
            ),
            (
                "Hedge Beta",
                hedge_beta,
                "number",
            ),
            (
                "Beta Reduction",
                beta_reduction,
                "percent",
            ),
            (
                "Tracking Error",
                te,
                "percent",
            ),
            (
                "Information Ratio",
                ir,
                "number",
            ),
        ]

        print(
            f"{'Metric':<35}"
            f"{'Value':>20}"
        )

        print("-" * width)

        for metric, value, fmt in effectiveness_rows:

            if fmt == "percent":
                value = fmt_percent(value)
            else:
                value = fmt_number(value)

            print(
                f"{metric:<35}"
                f"{value:>20}"
            )

        # ---------------------------------------------------------------------
        # Interpretation
        # ---------------------------------------------------------------------

        print()
        print("-" * width)
        print("INTERPRETATION")
        print("-" * width)

        if not pd.isna(volatility_reduction):

            if volatility_reduction > 0:
                print(
                    f"✓ Volatility reduced by "
                    f"{volatility_reduction:.2%}."
                )
            else:
                print(
                    f"✗ Volatility increased by "
                    f"{abs(volatility_reduction):.2%}."
                )

        if not pd.isna(drawdown_reduction):

            if drawdown_reduction > 0:
                print(
                    f"✓ Maximum drawdown reduced by "
                    f"{drawdown_reduction:.2%}."
                )
            else:
                print(
                    f"✗ Maximum drawdown became "
                    f"{abs(drawdown_reduction):.2%} worse."
                )

        if not pd.isna(downside_risk_reduction):

            if downside_risk_reduction > 0:
                print(
                    f"✓ Downside risk reduced by "
                    f"{downside_risk_reduction:.2%}."
                )
            else:
                print(
                    f"✗ Downside risk increased by "
                    f"{abs(downside_risk_reduction):.2%}."
                )

        if not pd.isna(return_difference):

            if return_difference > 0:
                print(
                    f"✓ Hedging increased annualized return by "
                    f"{return_difference:.2%}."
                )
            else:
                print(
                    f"• Hedging reduced annualized return by "
                    f"{abs(return_difference):.2%}."
                )

        if not pd.isna(hedge_beta):

            print(
                f"• Hedge beta: {hedge_beta:.3f}. "
                f"The hedged portfolio retains approximately "
                f"{hedge_beta:.1%} of the unhedged portfolio's "
                f"sensitivity to its own returns."
            )

        if not pd.isna(return_correlation):

            print(
                f"• Return correlation: "
                f"{return_correlation:.3f}."
            )

        # ---------------------------------------------------------------------
        # Overall assessment
        # ---------------------------------------------------------------------

        print()
        print("-" * width)
        print("OVERALL ASSESSMENT")
        print("-" * width)

        risk_improvements = 0

        if volatility_reduction > 0:
            risk_improvements += 1

        if downside_risk_reduction > 0:
            risk_improvements += 1

        if drawdown_reduction > 0:
            risk_improvements += 1

        if cvar_reduction > 0:
            risk_improvements += 1

        if risk_improvements >= 3:

            print(
                "✓ The hedge materially improved the portfolio's "
                "risk characteristics."
            )

        elif risk_improvements >= 2:

            print(
                "✓ The hedge provided a meaningful reduction "
                "in portfolio risk."
            )

        elif risk_improvements == 1:

            print(
                "• The hedge provided limited risk reduction."
            )

        else:

            print(
                "✗ The hedge did not materially improve the "
                "measured risk characteristics."
            )

        if (
            not pd.isna(return_difference)
            and return_difference < 0
            and volatility_reduction > 0
        ):

            print(
                "• The key trade-off is lower return in exchange "
                "for lower risk."
            )

        print()
        print("=" * width)
        print()

    # =========================================================================
    # Plot
    # =========================================================================

    def plot_results():

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 9),
            sharex=True
        )

        # ---------------------------------------------------------------------
        # Cumulative returns
        # ---------------------------------------------------------------------

        axes[0].plot(
            unhedged_wealth.index,
            unhedged_wealth,
            label="Unhedged",
            linewidth=2,
        )

        axes[0].plot(
            hedged_wealth.index,
            hedged_wealth,
            label="Hedged",
            linewidth=2,
        )

        axes[0].set_title(
            "Cumulative Portfolio Performance"
        )

        axes[0].set_ylabel(
            "Growth of $1"
        )

        axes[0].legend()

        axes[0].grid(
            alpha=0.3
        )

        # ---------------------------------------------------------------------
        # Drawdowns
        # ---------------------------------------------------------------------

        axes[1].plot(
            unhedged_dd.index,
            unhedged_dd,
            label="Unhedged",
            linewidth=2,
        )

        axes[1].plot(
            hedged_dd.index,
            hedged_dd,
            label="Hedged",
            linewidth=2,
        )

        axes[1].fill_between(
            unhedged_dd.index,
            unhedged_dd,
            0,
            alpha=0.10,
        )

        axes[1].set_title(
            "Portfolio Drawdown"
        )

        axes[1].set_ylabel(
            "Drawdown"
        )

        axes[1].legend()

        axes[1].grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.show()

    # =========================================================================
    # Execute output
    # =========================================================================

    if print_report:
        print_report()

    if plot:
        plot_results()

    return metrics_df, hedge_df
