import pandas as pd
import matplotlib.pyplot as plt
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
            return_type: str = "log",
            hac: bool = False,
            hac_lag: int = None

        ):
        """
        Parameters:
            asset1: ticker of independent variable asset (string)
            assets: list of tickers of dependent variable assets (list)
            period: optional - the data lookback period (string)
            frequency: frequency of returns data (string)
            start_date: optional - start date of data (string in YYYY-MM-DD)
            end_date: optional - end date of data (string in YYYY-MM-DD)
            return_type: optional - choose between "simple" returns definition or "log" returns definition (string)
            hac: optional - choose to regress with Heteroskedasticity-Autocorrelation Robust Covariance (boolean)
            hac_lag: optional - choose the maxlags for hac (integer)
        
        
        """
        
        if len(assets) < 1:
            raise ValueError("provide at least one factor asset")
        if return_type not in ("log", "simple"):
            raise ValueError("return_type = 'log' or return_type = 'simple' only.")
 
        self.asset1 = asset1
        self.freq = frequency
        
        self.return_type = return_type
 
        returns_fn = log_returns if return_type == "log" else simple_returns


        # hac : heteroscedasticity and autocorrelation robust (HAC) using n lags
        # users can override their hac lag number, but default will be automatic
        if hac is True:
            self.hac = True
            if hac_lag is not None:
                if type(hac_lag) == int:
                    self.hac_lags = self._resolve_hac_lags(hac_lag)
                else:
                    raise ValueError('hac_lag has to be integer!')
            elif hac_lag is None:
                self.hac_lags = self._resolve_hac_lags('auto')

        else:
            self.hac = False



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

        if self.hac:
            regress_obj = MultiFactorRegression(
                asset1=asset_1_returns,
                assets=assets_returns,
                return_type=return_type,
                hac=True,
                hac_lags = self.hac_lags
                )
        else:
            regress_obj = MultiFactorRegression(
                asset1=asset_1_returns,
                assets=assets_returns,
                return_type=return_type
                )


        # Store the multifactorregression obj as an attribute
        self.regress_obj = regress_obj


        # Store the list of independent variable assets that is actually used after combining timestamps
        self.merged_assets_names = regress_obj.x_col

        # Store the final merged return dataframe
        self.merged_return_series = regress_obj.merged_df.copy()

        self.n_obs = len(regress_obj.merged_df)

        # Store start date of regression
        self.start_date = regress_obj.start_date


        # Store end date of regression
        self.end_date = regress_obj.end_date
        

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
            f"{'Use HAC':<30}: {self.hac}",
            f"{'Hac lags':<30}: {self.hac_lags if self.hac else 'None'}",
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

        if not self.hac:
            self._diagnostics()


    def plot_results(self):
        factor_beta_and_ci_ax = mutlifac_ols_plot(self)
        plt.show()


    def historical_rolling_beta(self, window=60):

        if window <= 0:
            raise ValueError(
                f"observation_window must be positive, got {window}."
            )

        if len(self.regress_obj.merged_df) < window:
            raise ValueError(
                f"Insufficient data for rolling beta: "
                f"observation_window={window}, "
                f"but only {len(self.regress_obj.merged_df)} observations are available."
            )
    
        y = self.regress_obj.y_series
        X = self.regress_obj.x_series.copy()

        X = sm.add_constant(X, has_constant="add")


        window = window
        self.rolling_window = window

        rols = RollingOLS(
            endog=y,
            exog=X, 
            window=window
        )

        results = rols.fit()

        df = self.regress_obj.merged_df
        x_col = self.regress_obj.x_col

        
        params = results.params

        self.beta_series = params

        # Convert the NumPy result arrays back into DataFrames
        bse = pd.DataFrame(
            results.bse,
            index=params.index,
            columns=params.columns
        )

        tvalues = pd.DataFrame(
            results.tvalues,
            index=params.index,
            columns=params.columns
        )

        pvalues = pd.DataFrame(
            results.pvalues,
            index=params.index,
            columns=params.columns
        )

        rsquared = pd.Series(
            results.rsquared,
            index=params.index
        )


        ci = results.conf_int(alpha=0.05)

        rolling_dfs = {}

        for ticker in x_col:

            rolling_df = pd.DataFrame({
                'date': df['date'].values,
                'beta': params[ticker].values,
                'beta_std_error': bse[ticker].values,
                'beta_tstat': tvalues[ticker].values,
                'beta_pvalue': pvalues[ticker].values,
                'alpha': params['const'].values,
                'alpha_stf_error': bse['const'].values,
                'alpha_tstat': tvalues['const'].values,
                'alpha_pvalue': pvalues['const'].values,
                'r_squared': rsquared.values
            })

            # Confidence intervals
            rolling_df["beta_ci_lower"] = (ci[(ticker, "lower")].values)
            rolling_df["beta_ci_upper"] = (ci[(ticker, "upper")].values)

            rolling_df["alpha_ci_lower"] = (ci[("const", "upper")].values)
            rolling_df["alpha_ci_upper"] = (ci[("const", "upper")].values)

            # Annualized alpha
            if self.return_type == "simple":
                rolling_df["annualized_alpha"] = (
                    (1 + rolling_df["alpha"]) ** 252 - 1
                )
            else:
                rolling_df["annualized_alpha"] = (
                    rolling_df["alpha"] * 252
                )

            rolling_dfs[ticker] = rolling_df



        self.rolling_dfs = rolling_dfs

        return rolling_dfs


    def rolling_beta_summary(self):

        print("=" * 60)
        print(f"Rolling Beta Summary ({self.rolling_window}-Observations Window)")
        print("=" * 60)

        print(f"\nObservation period")
        print(f"  Start:              {self.regress_obj.start_date}")
        print(f"  End:                {self.regress_obj.end_date}")
        print(f"  Observations:       {self.n_obs}")
        print("\n")

        for ticker, rolling_df in self.rolling_dfs.items():

            latest = rolling_df.iloc[-1]

            print("=" * 60)
            print(f"Stats for constituent independent variable {ticker}-returns")
            print("=" * 60)

            print(f"\nCurrent estimates")
            print(f"  Beta:               {float(latest['beta']):.5f}")
            print(
                f"  95% CI:             "
                f"[{float(latest['beta_ci_lower']):.5f}, "
                f"{float(latest['beta_ci_upper']):.5f}]"
            )
            print(f"  Alpha:              {float(latest['alpha']):.6f}")
            print(f"  R-squared:          {float(latest['r_squared']):.3f}")
            #print(f"  Residual volatility: {float(latest['residual_volatility'].item()):.6f}")

            print(f"\nBeta significance")
            print(f"  Standard error:     {float(latest['beta_std_error']):.4f}")
            print(f"  t-statistic:        {float(latest['beta_tstat']):.2f}")
            print(f"  p-value:            {float(latest['beta_pvalue']):.4g}")

            print(f"\nBeta history")
            print(f"  Mean:               {float(rolling_df['beta'].mean()):.5f}")
            print(f"  Median:             {float(rolling_df['beta'].median()):.5f}")
            print(f"  Minimum:            {float(rolling_df['beta'].min()):.5f}")
            print(f"  Maximum:            {float(rolling_df['beta'].max()):.5f}")
            print(f"  Std. deviation:     {float(rolling_df['beta'].std()):.5f}")

            print(f"\nR-squared history")
            print(f"  Mean:               {float(rolling_df['r_squared'].mean()):.3f}")
            print(f"  Minimum:            {float(rolling_df['r_squared'].min()):.3f}")
            print(f"  Maximum:            {float(rolling_df['r_squared'].max()):.3f}")

            print("=" * 60)


    def rolling_beta_plot(self):

        fig, ax = plt.subplots(figsize=(12, 6))
        

        for ticker, df in self.rolling_dfs.items():

            df = df.copy()

            # Plot the beta trend line
            (line,) = ax.plot(
                df['date'],
                df['beta'],
                label = f'Rolling beta of {ticker}',
                linewidth = 1.6,
                alpha = 1,
            )

            # Plot the beta confidence intervals
            ax.fill_between(
                df['date'],
                df['beta_ci_lower'],
                df['beta_ci_upper'],
                color = line.get_color(),
                alpha = 0.2,
                label = f'{ticker} 95% Confidence interval'
            )

        ax.axhline(
            y=1,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.4,
            label="Beta = 1"
        )

        ax.axhline(
            y=0,
            color="black",
            linestyle=":",
            linewidth=1,
            alpha=0.4
        )
        
        ax.set_title(f"{self.rolling_window}-observation rolling beta")
        ax.set_xlabel("Date")
        ax.set_ylabel("Beta")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.gcf().autofmt_xdate()  # Rotates dates automatically for readability
        ax.legend()

        plt.tight_layout()
        plt.show()

        return fig


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


    def _resolve_hac_lags(self, hac="auto"):
        if hac is None:
            return None

        if isinstance(hac, int):
            if hac < 0:
                raise ValueError("HAC lags must be non-negative.")
            return hac

        if hac != "auto":
            raise ValueError(
                "hac must be 'auto', None, or a non-negative integer."
            )

        defaults = {
            "daily": 3,
            "weekly": 3,
            "monthly": 3,
            "annually": 1,
        }

        return defaults[self.freq]



# Example usage
if __name__ == "__main__":
    my_beta = MultiAssetsRegression(
        asset1="tsla",
        assets=['msft','aapl','goog','ko'],
        period='10y',
        hac=True,
        frequency='daily',
        return_type='log'
        )
    my_beta.summary()
    my_beta.plot_results
    my_beta.historical_rolling_beta(window=60)
    my_beta.rolling_beta_summary()
    my_beta.rolling_beta_plot()