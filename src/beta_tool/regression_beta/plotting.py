import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm, gaussian_kde, probplot
from matplotlib.lines import Line2D

def price_series_plot(ticker:str, data: pd.DataFrame, data_col=None, ax = None):
    """
    this function plots the time series of the asset price

    function accepts a pandas dataframe object and plots the graph
    
    """

    df = data.copy()

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    if not "date" in data.columns:
        raise KeyError("Missing 'date' column in dataframe provided!")
    
    if data_col is None:
        raise ValueError("Please set column name for data to be plotted")

    ax.plot(
        df["date"],
        df[data_col],
        label="adjusted-close price"
    )

    ax.set_title(
        f"Price series of {ticker}",
        fontsize=16,
        pad = 15
        )
    
    ax.legend()
    ax.grid(alpha=0.2)

    return ax


def returns_distribution_plot(ticker:str, data: pd.DataFrame, return_type:str="log", data_col:str = None, axes = None):
    """
    
    this function plots the distribution histogram/KDE of the asset returns, compared to a normal distribution with the empirical data parameters, and also a Q-Q plot

    it accepts a beta object or a pandas dataframe object
    
    """
    
    if data_col is None:
        raise ValueError("Please set column name for data to be plotted")
        
        
    df = data.copy()
    r = df[data_col]
    
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
    ax1.set_title(f"Distribution of {ticker} {return_type} Returns")
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


def two_asset_ols_plot(dependent_ticker:str, independent_ticker:str, dependent_asset_df: pd.DataFrame, independent_asset_df: pd.DataFrame, dependent_col, independent_col, return_type:str = "log", ax = None):

    """

    input two return series objects and plot their points on a scatter plot with OLS regression line visualisation

    """

    df1 = dependent_asset_df.copy()
    df2 = independent_asset_df.copy()

    #merge the two df by timeframes with how=inner so that the returns series will start on the latest timestamps which both assets have
    merged = pd.merge(
        df1,
        df2,
        on="date",
        how="inner",
        suffixes=("_1", "_2")
        )
    
    #hiding the time part of the pd datetime object
    merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')

    y = merged[dependent_col]
    X = merged[independent_col]

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
    ax.set_xlabel(f"{independent_ticker} {return_type} returns")
    ax.set_ylabel(f"{dependent_ticker} {return_type} returns")

    alpha = model.params['const']
    beta = model.params[independent_col]
    beta_pvalue = model.pvalues[independent_col]
    beta_tstat = model.tvalues[independent_col]
    beta_std_error = model.bse[independent_col]
    beta_ci_low = model.conf_int().loc[independent_col, 0]
    beta_ci_high = model.conf_int().loc[independent_col, 1]
    n = int(model.nobs)

    start_date = merged['date'].iloc[0]
    end_date = merged['date'].iloc[-1]

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


def beta_obj_ols_plot(beta_obj, ax=None):
    
    """

    input a Beta object and plot the points on a scatter plot with OLS regression line visualisation

    """


    model = beta_obj.olsmodel
    
    intercept, slope = model.params

    df1 = beta_obj.asset_1_returns.copy()
    df2 = beta_obj.asset_2_returns.copy()
    
    #merge the two df by timeframes with how=inner so that the returns series will start on the latest timestamps which both assets have
    merged = pd.merge(
        df1,
        df2,
        on="date",
        how="inner"
        )
        
    
    y = merged[beta_obj.y_col]
    X = merged[beta_obj.x_col]
    
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
    ax.set_xlabel(f"{beta_obj.asset2} {beta_obj.return_type} returns")
    ax.set_ylabel(f"{beta_obj.asset1} {beta_obj.return_type} returns")

    stats_df = beta_obj.ols_df.copy()
    
    stats_text = (
        f"$y = {stats_df['alpha'].item():.4f} + {stats_df['beta'].item():.4f}x$\n"
        f"$\\beta = {stats_df['beta'].item():.4f}$\n"
        f"$R^2$ = {stats_df['r_squared'].item():.4f}\n"
        f"$p(\\beta)$ = {stats_df['beta_pvalue'].item():.4g}\n"
        f"$N$ = {int(stats_df['n_obs'].item())}\n"
        f"$(\\beta)$ t-stat = {stats_df['beta_tstat'].item():.5f}\n"
        f"$(\\beta)$ std-error = {stats_df['beta_std_error'].item():.5f}\n"
        f"$(\\beta)$ confidence interval high = {stats_df['beta_ci_high'].item():.5f}\n"
        f"$(\\beta)$ confidence interval low = {stats_df['beta_ci_low'].item():.5f}\n"
        f"Start date: {stats_df['start_date'].item()}\n"
        f"End date: {stats_df['end_date'].item()}"
    )

    ax.text(
        0.05, 0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox = dict(boxstyle = "round", facecolor="white", alpha=0.8)
    )
    
    return ax


def mutlifac_ols_plot(multifac_obj, sort=True, ax=None):

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
    # Title and subtitle
    # ---------------------------------------------------------

    full_title = (
        f"{multifac_obj.asset1} — Multi-Factor Beta\n"
        f"n = {multifac_obj.regress_obj.observations:,}, R² = {multifac_obj.regress_obj.r_squared:.2f}\n"
        f"Start date = {multifac_obj.regress_obj.start_date}, End date = {multifac_obj.regress_obj.end_date}"
    )

    # Use pad to create space between the title and the plot line
    ax.set_title(full_title, fontdict={'fontsize': 12, 'fontweight': 'regular'}, pad=15)
    
    

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

    #Leave space for title and bottom information/legend
    fig.subplots_adjust(
        left=0.15,
        right=0.85,
        top=0.85,
        bottom=0.2,
    )

    return ax
