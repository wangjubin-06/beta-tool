import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from beta import Beta
from multibeta import MultiFactorRegression
from scipy.stats import norm, gaussian_kde, probplot
from matplotlib.lines import Line2D

def price_series_plot(data: pd.DataFrame, ax = None):
    """
    this function plots the time series of the asset price

    function accepts a pandas dataframe object and plots the graph
    
    """

    df = data.copy()

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))



    ax.plot(
        df["timestamp"],
        df["adjusted_close"],
        label="adjusted-close price"
    )

    ax.set_title(
        f"Price series of {df['symbol'].loc[1]}",
        fontsize=16,
        pad = 15
        )
    
    ax.legend()
    ax.grid(alpha=0.2)

    return ax

def returns_distribution_plot(data: pd.DataFrame, return_type:str, axes = None):
    """
    
    this function plots the distribution histogram/KDE of the asset returns, compared to a normal distribution with the empirical data parameters, and also a Q-Q plot

    it accepts a pandas dataframe object
    
    """
    df = data.copy()
    r = df[f"{return_type}-returns"]

    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(13, 5))

    ax1, ax2 = axes

    # Histogram as density
    ax1.hist(
        r,
        bins="fd",
        density=True,
        alpha=0.35,
        color="steelblue",
        edgecolor="white",
        label=f"Returns"
    )

    # KDE
    x = np.linspace(r.min(), r.max(), 500)
    kde = gaussian_kde(r)
    ax1.plot(x, kde(x), color="steelblue", lw=2, label="KDE")

    # Fitted normal distribution
    mu, sigma = r.mean(), r.std()
    ax1.plot(
        x,
        norm.pdf(x, mu, sigma),
        color="crimson",
        lw=1,
        linestyle="--",
        label=f"Normal\n($\\mu$={mu:.4f},\n$\\sigma$={sigma:.4f})"
    )

    ax1.set_xlabel(f"{return_type} Return")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Distribution of {df["symbol"].iloc[0]} {return_type} Returns")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    # --------------------
    # Q-Q plot
    # --------------------
    probplot(r, dist="norm", plot=ax2)

    ax2.set_title("Normal Q-Q Plot")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.grid(alpha=0.2)

    return axes



def two_asset_ols_plot(asset1: pd.DataFrame, asset2: pd.DataFrame, return_type:str = "log", ax = None):

    """

    input two return series objects and plot their points on a scatter plot with OLS regression line visualisation

    """

    df1 = asset1.copy()
    df2 = asset2.copy()

    #merge the two df by timeframes with how=inner so that the returns series will start on the latest timestamps which both assets have
    merged = pd.merge(
        df1,
        df2,
        on="timestamp",
        how="inner",
        suffixes=("_1", "_2")
        )
    
    #hiding the time part of the pd datetime object
    merged['timestamp'] = merged['timestamp'].dt.strftime('%Y-%m-%d')

    y = merged[f'{return_type}-returns_1']
    X = merged[f'{return_type}-returns_2']

    #OLS: y = alpha + beta * X
    X_with_constant = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_with_constant).fit()

    intercept, slope = model.params
    r_squared = model.rsquared

    #regression line
    x_line = np.linspace(X.min(), X.max(),100)
    y_line = intercept + slope * x_line

    #plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    

    ax.scatter(
        X,
        y,
        alpha = 0.7,
        s=30,
        label = f"Observed {return_type}-returns"
    )

    ax.plot(x_line, y_line, color = 'red', linewidth = 2, label = "OLS regression")

    ax.set_title('Scatter plot of two asset return series with OLS regression line')
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_xlabel(f"{df2['symbol'].iloc[1]} {return_type} returns")
    ax.set_ylabel(f"{df1['symbol'].iloc[1]} {return_type} returns")

    alpha = model.params['const']
    beta = model.params[f"{return_type}-returns_2"]
    beta_pvalue = model.pvalues[f"{return_type}-returns_2"]
    beta_tstat = model.tvalues[f"{return_type}-returns_2"]
    beta_std_error = model.bse[f"{return_type}-returns_2"]
    beta_ci_low = model.conf_int().loc[f"{return_type}-returns_2", 0]
    beta_ci_high = model.conf_int().loc[f"{return_type}-returns_2", 1]
    n = int(model.nobs)

    start_date = merged['timestamp'].iloc[0]
    end_date = merged['timestamp'].iloc[-1]

    stats_text = (
        f"$y = {float(alpha):.4f} + {float(beta):.4f}x$\n"
        f"$\\beta = {float(beta):.4f}$\n"
        f"$R^2$ = {float(r_squared):.4f}\n"
        f"$p(\\beta)$ = {float(beta_pvalue):.4g}\n"
        f"$N$ = {n}\n"
        f"$(\\beta)$ t-stat = {float(beta_tstat):.5f}\n"
        f"$(\\beta)$ std-error = {float(beta_std_error):.5f}\n"
        f"$(\\beta)$ confidence interval high = {float(beta_ci_high):.5f}\n"
        f"$(\\beta)$ confidence interval low = {float(beta_ci_low):.5f}\n"
        f"Start date: {start_date}\n"
        f"End date: {end_date}"
    )
    ax.text(
        0.05, 0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox = dict(boxstyle = "round", facecolor="white", alpha=0.8)
    )

    return ax


def beta_obj_ols_plot(beta_obj: Beta, ax=None):
    
    """

    input a Beta object and plot the points on a scatter plot with OLS regression line visualisation

    """


    model = beta_obj.olsresults
    
    intercept, slope = model.params

    df1 = beta_obj.asset_1_returns.copy()
    df2 = beta_obj.asset_2_returns.copy()
    
    #merge the two df by timeframes with how=inner so that the returns series will start on the latest timestamps which both assets have
    merged = pd.merge(
        df1,
        df2,
        on="timestamp",
        how="inner",
        suffixes=("_1", "_2")
        )
        
    
    y = merged[f'{beta_obj.return_type}-returns_1']
    X = merged[f'{beta_obj.return_type}-returns_2']
    
    #regression line
    x_line = np.linspace(X.min(), X.max(),100)
    y_line = intercept + slope * x_line
    
    #plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.scatter(
        X,
        y,
        alpha = 0.7,
        s=30,
        label = f"Observed {beta_obj.return_type}-returns"
        )
    
    ax.plot(x_line, y_line, color = "red", linewidth = "2", label = "OLS regression")
    
    ax.set_title('Scatter plot of two asset return series with OLS regression line')
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_xlabel(f"{df2['symbol'].iloc[1]} {beta_obj.return_type} returns")
    ax.set_ylabel(f"{df1['symbol'].iloc[1]} {beta_obj.return_type} returns")

    
    stats_text = (
        f"$y = {beta_obj.intercept:.4f} + {beta_obj.beta:.4f}x$\n"
        f"$\\beta = {float(beta_obj.beta):.4f}$\n"
        f"$R^2$ = {beta_obj.rsquare:.4f}\n"
        f"$p(\\beta)$ = {beta_obj.beta_p_value:.4g}\n"
        f"$N$ = {beta_obj.observations}\n"
        f"$(\\beta)$ t-stat = {beta_obj.beta_tstat:.5f}\n"
        f"$(\\beta)$ std-error = {beta_obj.beta_std_error:.5f}\n"
        f"$(\\beta)$ confidence interval high = {beta_obj.beta_ci_high:.5f}\n"
        f"$(\\beta)$ confidence interval low = {beta_obj.beta_ci_low:.5f}\n"
        f"Start date: {beta_obj.start_date}\n"
        f"End date: {beta_obj.end_date}"
    )

    ax.text(
        0.05, 0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox = dict(boxstyle = "round", facecolor="white", alpha=0.8)
    )
    
    return ax



def mutlifac_ols_plot(multifac_obj: MultiFactorRegression, sort=True, ax=None):

    """

    Plot factor loadings from a MultiFactorRegression with 95% confidence intervals.
    
    """

    betas = multifac_obj.betas

    assets = list(betas.keys())

    if sort:
        assets = sorted(
            assets,
            key=lambda x: betas[x]["beta"],
            reverse=True,
        )

    beta = np.array([betas[x]["beta"] for x in assets])
    ci_low = np.array([betas[x]["ci_low"] for x in assets])
    ci_high = np.array([betas[x]["ci_high"] for x in assets])
    p_values = np.array([betas[x]["p_value"] for x in assets])

    y = np.arange(len(assets))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # ---------------------------------------------------------
    # Betas + confidence intervals
    # ---------------------------------------------------------

    errorbar = ax.errorbar(
        beta,
        y,
        xerr=[
            beta - ci_low,
            ci_high - beta,
        ],
        fmt="o",
        capsize=4,
        label="Beta",
    )


    # ---------------------------------------------------------
    # Axes
    # ---------------------------------------------------------

    ax.set_yticks(y)
    ax.set_yticklabels(assets)

    ax.set_xlabel("Beta / Factor Loading")
    ax.set_ylabel("Factor", labelpad=20)

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.6,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", length=0)

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    ax.set_title(
        f"{multifac_obj.asset1} — Multi-Factor Beta",
        fontsize=18,
        pad=20,
    )

    # ---------------------------------------------------------
    # Error-bar annotations
    # ---------------------------------------------------------

    for i, (b, low, high, p) in enumerate(
        zip(beta, ci_low, ci_high, p_values)
    ):
        label = (
            f"β = {b:.2f}\n"
            f"95% CI [{low:.2f}, {high:.2f}]\n"
            f"p = {p:.2e}"
        )

        ax.annotate(
            label,
            xy=(high, i),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            alpha = 0.8,
        )

    # ---------------------------------------------------------
    # Bottom information
    # ---------------------------------------------------------

    bottom_info = (
        f"n = {multifac_obj.observations:,}    "
        f"R² = {multifac_obj.r_squared:.2f}    "
        f"Start date = {multifac_obj.start_date}    "
        f"End date = {multifac_obj.end_date}"
    )

    ax.text(
        0.5,
        0.04,
        bottom_info,
        ha="center",
        va="center",
    )

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------

    # Get the actual default color used by errorbar
    beta_color = errorbar.lines[0].get_color()

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=beta_color,
            label="Beta",
        ),
        Line2D(
            [0],
            [0],
            color=beta_color,
            label="95% confidence interval",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=True,
    )

    # Leave space for title and bottom information/legend
    # fig.subplots_adjust(
    #     left=0.15,
    #     right=0.85,
    #     top=0.85,
    #     bottom=0.2,
    # )

    return ax



if __name__ == "__main__":
    import data
    import returns
    import beta
    import multibeta

    # my_reg = beta.Beta(asset1="aapl", asset2="msft", period="5y", end_date="2026-02-03", return_type="log", interval="daily")

    # beta_obj_ols_plot(my_reg)

    # apple = data.AssetData(ticker="aapl",interval="daily",start_date = "2013-01-01", end_date = "2021-01-01")
    # msft = data.AssetData(ticker="msft",interval="daily",start_date = "2013-01-01", end_date = "2021-01-01")
    
    # aapleprices = apple.get_prices()
    # msftprices = msft.get_prices()

    # aaplreturns = returns.log_returns(aapleprices)
    # msftreturns = returns.log_returns(msftprices)

    #two_asset_ols_plot(aaplreturns, msftreturns, "log")

    # returns_distribution_plot(aaplreturns,"log")

    my_multi = multibeta.MultiFactorRegression(asset1="aapl", assets=["msft", "goog","nvda","ko","intc","spy","iwm"], period = "5y", end_date="2026-02-03", return_type="log", interval="daily")

    mutlifac_ols_plot(my_multi)
