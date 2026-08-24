from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from regression_beta.data import AssetData
from regression_beta.returns import log_returns, simple_returns
from regression_beta.regression import MultiFactorRegression
from regression_beta.plotting import mutlifac_ols_plot
from regression_beta.diagnostics import heteroskedasticity, autocorrelation, multicollinearity, normality
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
 
class MultiAssetsRegression:
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
            frequency: str = "daily",
            start_date: str | None = None,
            end_date: str | None = None,
            return_type: str = "log"
        ):
        
        if len(assets) < 1:
            raise ValueError("provide at least one factor asset")
        if return_type not in ("log", "simple"):
            raise ValueError("return_type = 'log' or return_type = 'simple' only.")
 
        self.asset1 = asset1
        self.freq = frequency
        
        self.return_type = return_type
 
        returns_fn = log_returns if return_type == "log" else simple_returns




        # Get dependent variable asset returns df
        asset_1_prices = AssetData(asset1, period, frequency, start_date, end_date).get_prices()
        asset_1_returns = returns_fn(asset_1_prices)



        # Pull the prices of independent variable assets and store in a dict
        assets_prices = {}

        for ticker in assets:

            prices = AssetData(ticker, period, frequency, start_date, end_date).get_prices()

            assets_prices[ticker.lower()] = prices

        

        # Calculate the returns df of independent variable assets and store it in a dict
        assets_returns = {}


        for ticker in assets:

            prices = assets_prices[ticker.lower()]


            assets_returns[ticker.upper()] = returns_fn(prices)

        


        # Create the multifactorregression object
        regress_obj = MultiFactorRegression(
            asset_1_returns,
            assets_returns,
            return_type=return_type
            )


        # Store the multifactorregression obj as an attribute
        self.regress_obj = regress_obj


        # Store the list of independent variable assets that is actually used after combining timestamps
        self.merged_assets_names = regress_obj.x_col

        # Store the final merged return dataframe
        self.merged_return_series = regress_obj.merged_return_series.copy()

        # Store start date of regression
        self.start_date = regress_obj.merged_return_series['date'].iloc[0]


        # Store end date of regression
        self.end_date = regress_obj.merged_return_series['date'].iloc[-1]
        

        # Model of the OLS
        results = regress_obj.ols()


        # Store the model
        self.olsresults = results


        # Store the stats as attributes
        conf_int = results.conf_int()
 
        self.intercept = float(results.params["const"])
        self.r_squared = float(results.rsquared)
        self.observations = int(results.nobs)
        self.residual_vol = float(results.resid.std())
 
        # per-asset stats, keyed by ticker, mirroring Beta's single-asset attributes
        self.betas = {}
        for name in regress_obj.x_col:
            self.betas[name] = {
                "beta": float(results.params[name]),
                "p_value": float(results.pvalues[name]),
                "t_stat": float(results.tvalues[name]),
                "std_error": float(results.bse[name]),
                "ci_low": float(conf_int.loc[name, 0]),
                "ci_high": float(conf_int.loc[name, 1]),
            }



    # Public APIs
    def summary(self) -> str:

        lines = [
            '\n\n=====================================================',
            f"Multi-Factor OLS Regression: {self.asset1} against {', '.join(self.merged_assets_names)}:",
            '=====================================================',
            f"{'Intercept':<30}: {self.intercept:.5f}",
            f"{'R-squared':<30}: {self.r_squared:.5f}",
            f"{'Start date':<30}: {self.start_date}",
            f"{'End date':<30}: {self.end_date}",
            f"{'Frequency':<30}: {self.freq}",
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
        print("\n".join(lines))
        self._diagnostics()


    def plot_results(self):
        factor_beta_and_ci_ax = mutlifac_ols_plot(self)
        plt.show()


    # Private methods
    def __str__(self) -> str:
        return self.summary()

    def _diagnostics(self):
        """This prints the results analysis on the regression results.
        """

        heteroskedasticity(self.olsresults)
        
        autocorrelation(self.olsresults)
        
        normality(self.olsresults)
        
        multicollinearity(self.olsresults)
        
        print('\n\n\n')


# Example usage
if __name__ == "__main__":
    my_beta = MultiAssetsRegression("tsla", ['msft','aapl'], '5y')
    my_beta.summary()