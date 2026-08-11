import data, regression, returns
from datetime import date


class Beta():

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

        self.asset1 = asset1.upper()
        self.asset2 = asset2.upper()
        self.return_type = return_type


        asset_1_prices = data.AssetData(asset1, period, interval, start_date, end_date).get_prices()
        asset_2_prices = data.AssetData(asset2, period, interval, start_date, end_date).get_prices()

        returns_fn = returns.log_returns if return_type == "log" else returns.simple_returns

        asset_1_returns = returns_fn(asset_1_prices)
        asset_2_returns = returns_fn(asset_2_prices)

        self.asset_1_returns = asset_1_returns
        self.asset_2_returns = asset_2_returns

        #asset_1_returns is dependent variable (Y), asset_2_returns is independent variable (X)

        regress_obj = regression.OLSRegression(
            asset_1_returns,
            asset_2_returns,
            return_type=return_type
            )

        self.start_date = regress_obj.merged_return_series["timestamp"].iloc[0]
        self.end_date = regress_obj.merged_return_series["timestamp"].iloc[-1]
        
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




if __name__ == "__main__":
    my_beta = Beta(asset1="msft", asset2="aapl", period="5y", interval="daily", return_type="log")
    print(my_beta.summary())