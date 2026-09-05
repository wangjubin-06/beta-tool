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
        
        self.assets = assets
        
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
                frequency= self.freq,
                hac=True,
                hac_lags = self.hac_lags
                )
        else:
            regress_obj = MultiFactorRegression(
                asset1=asset_1_returns,
                assets=assets_returns,
                return_type=return_type,
                frequency=self.freq
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
        
        self.betas = regress_obj.betas




    # Public APIs
    def summary(self) -> str:
        
        self.regress_obj.summary(asset_1_name=self.asset1)
        
        if not self.hac:
            self._diagnostics()


    def plot_results(self):
        
        factor_beta_and_ci_ax = mutlifac_ols_plot(self)
        plt.show()


    def historical_rolling_beta(self, window=60):

        rolling_dfs = self.regress_obj.rolling_ols(window=window)
        
        


    def rolling_beta_summary(self):

        self.regress_obj.rolling_beta_summary()
        


    def rolling_beta_plot(self):

        self.regress_obj.rolling_beta_plot()
        
        


    def get_beta(self):
        
        beta_dic = {}
        
        for ticker, dic in self.regress_obj.betas.items():
            beta_dic[ticker] = dic['beta']
            
        return beta_dic



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
    my_beta.plot_results()
    my_beta.historical_rolling_beta(window=60)
    my_beta.rolling_beta_summary()
    my_beta.rolling_beta_plot()