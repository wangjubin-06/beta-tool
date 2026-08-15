import data, beta_tool.regression_beta.regression as regression, beta_tool.regression_beta.returns as returns, beta_tool.regression_beta.beta as beta, beta_tool.regression_beta.plotting as plotting
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
 
 
class MultiAssetsRegression():
    """
    Generic multi-asset factor tool: regresses one asset's returns against N
    other assets' returns simultaneously. Same idea as Beta, just not limited
    to a single explanatory asset — the "factors" can be any tickers, not a
    fixed factor set like Fama-French.
    """
 
    def __init__(
            self,
            asset1: str, #the dependent asset on y-axis
            assets: list[str], #the list of assets for x-axis
            period: str = "1y",
            interval: str = "daily",
            start_date: date | None = None,
            end_date: date | None = None,
            return_type: str = "log"
            ):
        
        if len(assets) < 1:
            raise ValueError("provide at least one factor asset")
        if return_type not in ("log", "simple"):
            raise ValueError("return_type = 'log' or return_type = 'simple' only.")
 
        self.asset1 = asset1.upper()
        self.interval = interval
        
        self.return_type = return_type
 
        returns_fn = returns.log_returns if return_type == "log" else returns.simple_returns
 
        asset_1_prices = data.AssetData(asset1, period, interval, start_date, end_date).get_prices()
        asset_1_returns = returns_fn(asset_1_prices)

        self.asset_1_prices = asset_1_prices
        self.asset_1_returns = asset_1_returns

        assets_prices = {}
        for ticker in assets:
            prices = data.AssetData(ticker, period, interval, start_date, end_date).get_prices()
            assets_prices[ticker.upper()] = prices

        self.assets_prices = assets_prices

        assets_returns = {}
        for ticker in assets:
            prices = assets_prices[ticker.upper()]
            assets_returns[ticker.upper()] = returns_fn(prices)

        self.assets_returns = assets_returns
 
        regress_obj = regression.MultiFactorRegression(
            asset_1_returns,
            assets_returns,
            return_type=return_type
            )

        self.merged_assets_names = regress_obj.assets_names #this is a list of names of assets that are actually used in the end for the factor regression of asset1 against asset(s) after combining common timestamps available in the return series

        self.start_date = regress_obj.merged_return_series['timestamp'].iloc[0]
        self.end_date = regress_obj.merged_return_series['timestamp'].iloc[-1]
        #these are the start and end dates which includes all available data for all assets.
 
        results = regress_obj.ols()
        self.olsresults = results

        conf_int = results.conf_int()
 
        self.intercept = float(results.params["const"])
        self.r_squared = float(results.rsquared)
        self.observations = int(results.nobs)
        self.residual_vol = float(results.resid.std())
 
        # per-asset stats, keyed by ticker, mirroring Beta's single-asset attributes
        self.betas = {}
        for name in regress_obj.assets_names:
            self.betas[name] = {
                "beta": float(results.params[name]),
                "p_value": float(results.pvalues[name]),
                "t_stat": float(results.tvalues[name]),
                "std_error": float(results.bse[name]),
                "ci_low": float(conf_int.loc[name, 0]),
                "ci_high": float(conf_int.loc[name, 1]),
            }

 
    def summary(self) -> str:
        lines = [
            f"Multi-Factor OLS Regression: {self.asset1} against {', '.join(self.assets_tickers)}:",
            "-" * 60,
            f"{'Intercept':<30}: {self.intercept:.5f}",
            f"{'R-squared':<30}: {self.r_squared:.5f}",
            f"{'No. of observations':<30}: {self.observations}",
            f"{'Residual volatility':<30}: {self.residual_vol:.5f}",
            "",
        ]
        for name, stats in self.betas.items():
            lines.append(f"{name}:")
            lines.append(f"  {'Beta':<28}: {stats['beta']:.5f}")
            lines.append(f"  {'p-value':<28}: {stats['p_value']:.2e}")
            lines.append(f"  {'t-stat':<28}: {stats['t_stat']:.5f}")
            lines.append(f"  {'std-error':<28}: {stats['std_error']:.5f}")
            lines.append(f"  {'CI low':<28}: {stats['ci_low']:.5f}")
            lines.append(f"  {'CI high':<28}: {stats['ci_high']:.5f}")
        return "\n".join(lines)
 
    def __str__(self) -> str:
        return self.summary()

    def plot_results(self):
        factor_beta_and_ci_ax = plotting.multifac_ols_plot(self)
        plt.show()


 
 
if __name__ == "__main__":
    my_multi_beta = MultiAssetsRegression(
        asset1="tsla",
        assets=["aapl", "msft", "spy"],
        period="5y",
        interval="daily",
        return_type="log"
        )
    print(my_multi_beta.summary())