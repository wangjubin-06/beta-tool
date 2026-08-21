from regression_beta.data import AssetData
from regression_beta import regression
from regression_beta import plotting
from regression_beta.returns import log_returns, simple_returns
from datetime import date
import matplotlib.pyplot as plt
from regression_beta.rolling import historical_rolling_beta, historical_rolling_beta_plot


class Beta:

    def __init__(
            self,
            asset1: str, #the dependent asset on y-axis
            asset2: str, #the independent asset on x-axis
            period: str = "1y",
            interval: str = "daily",
            start_date: date | None = None,
            end_date: date | None = None,
            return_type: str = "log"
            ):
        
        if return_type not in ("log","simple"):
            raise ValueError("return_type = 'log' or return_type = 'simple' only.")

        self.asset1 = asset1
        self.asset2 = asset2
        self.return_type = return_type


        asset_1_prices = AssetData(
            ticker = asset1,
            period = period,
            frequency = interval,
            start_date = start_date,
            end_date = end_date
        ).get_prices()

        asset_2_prices = AssetData(
            ticker = asset2,
            period = period,
            frequency = interval,
            start_date = start_date,
            end_date = end_date
        ).get_prices()

        

        self.asset_1_prices = asset_1_prices
        self.asset_2_prices = asset_2_prices

        returns_fn = log_returns if return_type == "log" else simple_returns

        asset_1_returns = returns_fn(
            data = asset_1_prices,
            header_name="adjClose"
        )

        asset_2_returns = returns_fn(
            data = asset_2_prices,
            header_name="adjClose"
        )

        self.asset_1_returns = asset_1_returns
        self.asset_2_returns = asset_2_returns

        #asset_1_returns is dependent variable (Y), asset_2_returns is independent variable (X)

        regress_obj = regression.OLSRegression(
            asset_1_returns,
            asset_2_returns,
            return_type=return_type
            )
        
        self.merged_returns_series = regress_obj.merged_return_series.copy()
        
        self.merged_asset_1_returns = regress_obj.y.copy()
        self.merged_asset_2_returns = regress_obj.x.copy()

        self.start_date = regress_obj.merged_return_series["date"].iloc[0]
        self.end_date = regress_obj.merged_return_series["date"].iloc[-1]
        
        results = regress_obj.ols()
        self.olsresults = results

        stats = {
            "beta": results.params[f"{return_type}-returns_2"],
            "intercept": results.params['const'],
            "r_squared": results.rsquared,
            "beta_pvalue": results.pvalues[f"{return_type}-returns_2"],
            "beta_tstat": results.tvalues[f"{return_type}-returns_2"],
            "beta_std_error": results.bse[f"{return_type}-returns_2"],
            "beta_ci_low": results.conf_int().loc[f"{return_type}-returns_2", 0],
            "beta_ci_high": results.conf_int().loc[f"{return_type}-returns_2", 1],
            "n_obs": results.nobs,
            "residual_vol": results.resid.std()
            }

        self.beta = float(stats["beta"])
        self.intercept = float(stats["intercept"])
        self.rsquare = float(stats["r_squared"])
        self.beta_p_value = float(stats["beta_pvalue"])
        self.beta_tstat = float(stats["beta_tstat"])
        self.beta_std_error = float(stats["beta_std_error"])
        self.beta_ci_low = float(stats["beta_ci_low"])
        self.beta_ci_high = float(stats["beta_ci_high"])
        self.observations = int(stats["n_obs"])
        self.residual_vol = float(stats["residual_vol"])

    def summary(self) -> str:
        """Return a formatted summary of the regression results."""
        lines = [
            f"OLS Regression: {self.asset1} against {self.asset2}:",
            "-" * 50,
            f"{'Beta':<30}: {self.beta:.5f}",
            f"{'Intercept':<30}: {self.intercept:.5f}",
            f"{'R-squared':<30}: {self.rsquare:.5f}",
            f"{'Beta p-value':<30}: {self.beta_p_value:.4g}",
            f"{'Beta t-stat':<30}: {self.beta_tstat:.5f}",
            f"{'Beta std-error':<30}: {self.beta_std_error:.5f}",
            f"{'Beta Confidence Interval high':<30}: {self.beta_ci_high:.5f}",
            f"{'Beta Confidence Interval low':<30}: {self.beta_ci_low:.5f}",
            f"{'No. of observations':<30}: {self.observations}",
            f"{'Residual volatility':<30}: {self.residual_vol:.5f}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def plot_results(self):
        fig = plt.figure(figsize=(20,16), layout="constrained")

        gs = fig.add_gridspec(
            nrows=4,
            ncols=4,
            height_ratios=[1.1, 1.0, 1.4, 1.4],
        )

        ax_asset_1_price = fig.add_subplot(gs[0, :2])
        ax_asset_2_price = fig.add_subplot(gs[0, 2:4])
        ax_asset_1_dist = fig.add_subplot(gs[1, 0])
        ax_asset_1_qq = fig.add_subplot(gs[1, 1])
        ax_asset_2_dist = fig.add_subplot(gs[1, 2])
        ax_asset_2_qq = fig.add_subplot(gs[1, 3])
        
        ax_ols = fig.add_subplot(gs[2:4, :])

        plotting.price_series_plot(ticker = self.asset1, data=self.asset_1_prices, ax = ax_asset_1_price)
        plotting.price_series_plot(ticker = self.asset2, data=self.asset_2_prices, ax = ax_asset_2_price)

        plotting.returns_distribution_plot(ticker = self.asset1, data=self.asset_1_returns, axes = (ax_asset_1_dist, ax_asset_1_qq), return_type=self.return_type)
        plotting.returns_distribution_plot(ticker = self.asset2, data=self.asset_2_returns, axes = (ax_asset_2_dist, ax_asset_2_qq), return_type=self.return_type)

        plotting.beta_obj_ols_plot(beta_obj=self, ax=ax_ols)

        return fig

    def plot_historical_rolling_beta(self,observation_window: int = 60):

        attr_name = f"{observation_window}_day_rolling_beta_data"
        attr = historical_rolling_beta(self, observation_window)

        setattr(self,attr_name,attr)

        ax = historical_rolling_beta_plot(attr)

        plt.show()



if __name__ == "__main__":
    my_beta = Beta(asset1="msft", asset2="aapl", period="5y", interval="daily", return_type="log")
    my_beta.plot_results()
    my_beta.plot_historical_rolling_beta(observation_window=60)