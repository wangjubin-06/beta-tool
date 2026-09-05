import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from regression_beta.data import AssetData
from regression_beta.diagnostics import *
from regression_beta.plotting import returns_distribution_plot, two_asset_ols_plot
from regression_beta.returns import log_returns, simple_returns
from regression_beta.rolling import historical_rolling_beta, rolling_beta_plot, rolling_beta_summary
from regression_beta.regression import OLSRegression, MultiFactorRegression

class PortfolioBeta:
    """
    
    This class takes a portfolio of assets and finds the asset weighted returns
    and regresses it against a benchmark to find portfolio beta
    
    """
    
    ALLOWED_FREQUENCIES = {
        'daily',
        'weekly',
        'monthly',
        'annually'
    }
    


    def __init__(self, portfolio_dic: dict, asset_to_be_regressed: str | list = "spy", frequency = 'daily', period = '10y', start_date = None, end_date = None, return_type = 'simple', hac: bool = False, hac_lag: int | None = None):
        """
        Parameters:
        portfolio_dic

        type: dictionary
        key-value pairs mapping asset ticker name to its percentage

        example:
            portfolio_dic = {
                "aapl": 10,
                "msft": 70,
                "nvda": 20
            }

        """
        

        # Initial checks
        if len(portfolio_dic) < 1:
            raise ValueError("portfolio cannot be empty")
        
        if frequency not in self.ALLOWED_FREQUENCIES:
            raise ValueError("only daily, weekly, monthly, annually is allowed for data interval!")
        
        if return_type not in {'log','simple'}:
            raise ValueError("only 'simple' or 'log' returns allowed for 'return_type'!")


        self.freq = frequency
    
    
        # Cleaning up and checking the dictionary for consistent formatting
        
        cleaned_portfolio_dic = {}
        
        ticker_set = set()
        
        weight_sum = 0
        
        
        for ticker, weight in portfolio_dic.items():
            
            if not type(ticker) == str:
                raise ValueError("Asset ticker has to be a string! One or more tickers are not strings!")
            
            if not (type(weight) == float or type(weight) == int):
                raise ValueError("Asset weightage has to be in numbers. One or more weights you provided are not numbers!")
            
            if weight < 0 or weight > 100:
                raise ValueError("Each asset weightage has to be between 0% to 100%. One or more of them are not!")
            
            
            if ticker.lower() not in ticker_set:
                
                ticker_set.add(ticker.lower())
                
                cleaned_portfolio_dic[ticker.lower()] = float(weight)
                
                weight_sum += weight
                
            else:
                
                cleaned_portfolio_dic[ticker.lower()] += float(weight)
                
                weight_sum += weight
        
        
        if not np.isclose(weight_sum, 100.0):
            raise ValueError("sum of asset weights is not 100!")


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
                self.hac_lags = self._resolve_hac_lags()
        else:
            self.hac = False


        self.portfolio_dic = cleaned_portfolio_dic
        
        self.period = period
        
        self.start_date = start_date
        
        self.end_date = end_date
        
        self.return_type = return_type
        
        
        
        if type(asset_to_be_regressed) == str:
            self.independent = asset_to_be_regressed.lower()
            self.multi_independent_asset = False
        elif type(asset_to_be_regressed) == list:
            self.independent = asset_to_be_regressed
            self.multi_independent_asset = True
        
        
        portfolio_desc = ""
        
        for ticker, weight in self.portfolio_dic.items():
            portfolio_desc += f'{ticker} - '
            portfolio_desc += f'{weight:.4f} %\n'
        
        self.portfolio_desc = portfolio_desc
        
        
        # Getting relevant returns data
        portfolio_merged_df, independent_data = self._get_data()
        
        # Regression
        self._regress(portfolio_merged_df, independent_data)


    # Public APIs
    def summary(self):
        
        
        if not self.multi_independent_asset:
            """Return a formatted summary of the static regression results."""
            
            
            ols_obj = self.ols_obj
            
            ols_obj.summary(asset_1_name=self.portfolio_desc,asset_2_name=self.independent)

            # print("=" * 60)
            # print(f" OLS Regression Summary")
            # print(f"Portfolio consisting of \n{portfolio_desc}against {self.independent}")
            # print("=" * 60)

            # print(f"\nObservation period")
            # print(f"  Start:              {self.ols_df['start_date'].item()}")
            # print(f"  End:                {self.ols_df['end_date'].item()}")
            # print(f"  Observations:       {int(self.ols_df['n_obs'].item())}")
            # print(f"  Frequency:          {self.freq}")
            # print(f"  Return type:        {self.return_type}")
            # print(f"  Heteroskedasticity-Autocorrelation Robust Covariance:        {self.hac}")

            # if self.hac:
            #     print(f"  Heteroskedasticity-Autocorrelation Robust Covariance lags:        {self.hac_lags}")

            # print(f"  Beta:               {float(self.ols_df['beta'].item()):.5f}")
            # print(
            #     f"  95% CI:             "
            #     f"[{float(self.ols_df['beta_ci_low'].item()):.5f}, "
            #     f"{float(self.ols_df['beta_ci_high'].item()):.5f}]"
            # )
            # print(f"  Alpha:              {float(self.ols_df['alpha'].item()):.6f}")
            # print(f"  Annualized Alpha    {float(self.ols_df['annualized_alpha'].item()):.3f}")
            # print(f"  Alpha p-value       {float(self.ols_df['alpha_p_value'].item()):.4g}")
            # print(f"  R-squared:          {float(self.ols_df['r_squared'].item()):.3f}")
            # print(f"  Residual volatility: {float(self.ols_df['residual_volatility'].item()):.6f}")

            # print(f"\nBeta significance")
            # print(f"  Standard error:     {float(self.ols_df['beta_std_error'].item()):.4f}")
            # print(f"  t-statistic:        {float(self.ols_df['beta_tstat'].item()):.2f}")
            # print(f"  p-value:            {float(self.ols_df['beta_pvalue'].item()):.4g}")
            # print("\n")

            if not self.hac:
                self._diagnostics()
        
        elif self.multi_independent_asset:
            """Return a formatted summary of the static regression results."""
            
            portfolio_desc = ""
            
            for ticker, weight in self.portfolio_dic.items():
                portfolio_desc += f'{ticker}-'
                portfolio_desc += f'{weight:.4f} %\n'
            
            
            self.multi_regress_obj.summary(asset_1_name=portfolio_desc)

            if not self.hac:
                self._diagnostics()


    def plot_results(self):
        
        if not self.multi_independent_asset:
            """
            Plot the results of the static regression
            """
            fig = plt.figure(
                figsize=(14,10),
            )

            gs = fig.add_gridspec(
                nrows=2,
                ncols=4,
                height_ratios=[1.0, 1.5],
            )

            ax_asset_1_dist = fig.add_subplot(gs[0, 0])
            ax_asset_1_qq = fig.add_subplot(gs[0, 1])
            ax_asset_2_dist = fig.add_subplot(gs[0, 2])
            ax_asset_2_qq = fig.add_subplot(gs[0, 3])
            
            ax_ols = fig.add_subplot(gs[1, :])


            portfolio_data_col = 'portfolio_log_returns' if self.return_type == 'log' else 'portfolio_simple_returns'
            
            returns_distribution_plot(
                ticker = "Portfolio",
                data=self.portfolio_returns_data,
                axes = (ax_asset_1_dist, ax_asset_1_qq),
                return_type=self.return_type,
                data_col = portfolio_data_col
            )
            
            
            returns_distribution_plot(
                ticker = self.independent,
                data = self.independent_returns_data,
                axes = (ax_asset_2_dist, ax_asset_2_qq),
                return_type = self.return_type,
                data_col = 'independent_variable_returns'
            )


            two_asset_ols_plot(
                dependent_ticker = "Portfolio",
                independent_ticker = self.independent,
                dependent_asset_df = self.portfolio_returns_data,
                independent_asset_df = self.independent_returns_data,
                dependent_col = portfolio_data_col,
                independent_col = 'independent_variable_returns',
                return_type = self.return_type,
                ax = ax_ols
            )

            plt.show()
            return fig
        
        elif self.multi_independent_asset:
            try:
                # Intentionally raising an error
                raise ValueError("2D plotting for the linear regression is only available for regression of portfolio against one asset only, cannot plot for regression of portfolio against multiple assets!")
            except ValueError as e:
                # This block runs instead of crashing the script
                print(f"\nCaught an error: {e}\n")
    
            



    def historical_rolling_beta(self, window=60):
        """
        Initiate historical rolling OLS for the portfolio against benchmark to get rolling beta and statistics

        Parameters:
            window: integer (optional)

            Specifies the look-back window for the rolling statistics
            Defaults to 60 observations window
        
        """
        
        
        if not self.multi_independent_asset:
            
            self.rolling_window = window

            y_col = 'portfolio_simple_returns' if self.return_type == 'simple' else 'portfolio_log_returns'

            rolling_df = historical_rolling_beta(
                y_df=self.portfolio_returns_data.copy(),
                x_df=self.independent_returns_data.copy(),
                y_col= y_col,
                x_col='independent_variable_returns',
                return_type=self.return_type,
                window= window
            )

            
            self.rolling_df = rolling_df.copy()
        
        
        
        elif self.multi_independent_asset:
            
            self.multi_regress_obj.rolling_ols(window=window)
            
        
        return
            
        

    def rolling_beta_summary(self):
        """
        Print the summary of the historical rolling statistics
        """
        if not self.multi_independent_asset:
            rolling_beta_summary(
                rolling_df=self.rolling_df,
                window=self.rolling_window
            )
        
        elif self.multi_independent_asset:
            self.multi_regress_obj.rolling_beta_summary()
            
        return
            
        

    def rolling_beta_plot(self):
        """
        Visualise the rolling beta
        
        """
        if not self.multi_independent_asset:
            fig = rolling_beta_plot(
                rolling_df=self.rolling_df,
                window=self.rolling_window
            )
        
        elif self.multi_independent_asset:
            self.multi_regress_obj.rolling_beta_plot()

        return


    # Private methods
    def _regress(self, merged_df, independent_data):

        if self.return_type == 'log':
            y_col = 'portfolio_log_returns'
        else:
            y_col = 'portfolio_simple_returns'


        if type(self.independent) == str:
            x_col = 'independent_variable_returns'
            
            if self.hac:
                ols_obj = OLSRegression(
                    asset1=merged_df,
                    asset2=independent_data,
                    asset_1_col=y_col,
                    asset_2_col=x_col,
                    frequency=self.freq,
                    return_type=self.return_type,
                    hac=True,
                    hac_lags=self.hac_lags
                )
            else:
                ols_obj = OLSRegression(
                    asset1=merged_df,
                    asset2=independent_data,
                    asset_1_col=y_col,
                    asset_2_col=x_col,
                    frequency=self.freq,
                    return_type=self.return_type,
                )


            model = ols_obj.ols()
            self.olsresults = model
            
            

            ols_obj.results_df()

            #self.ols_df = ols_df
            
            self.ols_obj = ols_obj
        
        
        elif type(self.independent) == list:
            
            if self.hac:
                regress_obj = MultiFactorRegression(
                    asset1= merged_df,
                    assets= independent_data,
                    frequency= self.freq,
                    asset_1_col= y_col,
                    return_type= self.return_type,
                    hac=True,
                    hac_lags=self.hac_lags
                )
            else:
                regress_obj = MultiFactorRegression(
                    asset1= merged_df,
                    assets= independent_data,
                    frequency= self.freq,
                    asset_1_col= y_col,
                    return_type= self.return_type
                )
            
            model = regress_obj.ols()
            self.olsresults = model
            
            self.multi_regress_obj = regress_obj
            


    def _get_data(self):
        
        # Getting price data for each asset in portfolio
        prices_dict = {}
        
        for asset in self.portfolio_dic.keys():
            
            dataobj = AssetData(ticker = asset, period = self.period, frequency= self.freq, start_date= self.start_date, end_date=self.end_date)
            
            df = dataobj.get_prices()
            
            if df.empty:
                raise ValueError(f"No price data available for {asset}!")
            
            prices_dict[asset.lower()] = df
        
        
        # Calculating weighted returns data for each asset
        returns_dict = {}
        
        returns_fn = simple_returns
        
        col = "simple-returns"
        
        for asset, price_df in prices_dict.items():
            
            asset_returns_df = returns_fn(price_df)
            
            # Multiplying by the portfolio weight to get portfolio weighted return
            asset_returns_df[col] = asset_returns_df[col] * (self.portfolio_dic[asset] / 100)
            
            
            # Renaming the column for easier reference after merging 
            asset_returns_df = asset_returns_df.rename(columns={col: f'{asset}-{col}'})
            
            # Making sure date column has datetime objects
            asset_returns_df["date"] = (pd.to_datetime(asset_returns_df["date"]).dt.normalize())
            
            returns_dict[asset] = asset_returns_df[['date', f'{asset}-{col}']].copy()
        
        
        
        # Inner merge the returns-df based on date column
        returns_df_list = list(returns_dict.values())
        
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), returns_df_list)
        
        
        # Create a column in merged_df that is the sum of the returns of each asset
        merge_cols = [f'{asset}-{col}' for asset in returns_dict.keys()]
        
        merged_df['portfolio_simple_returns'] = merged_df[merge_cols].sum(axis=1)
        

        merged_df['portfolio_log_returns']  = np.log1p(merged_df['portfolio_simple_returns'])
        
        if self.return_type == 'simple':
            merged_df = merged_df[['date', 'portfolio_simple_returns']].copy()
        else:
            merged_df = merged_df[['date', 'portfolio_log_returns']].copy()
        
        
        self.portfolio_returns_data = merged_df
        
        
        # Get returns data for independent variable
        
        if type(self.independent) == str:
            independent = AssetData(self.independent, self.period, self.freq, self.start_date, self.end_date)
                        
            independent_df = independent.get_prices()
            
            ind_returns_fn = simple_returns if self.return_type == 'simple' else log_returns
            
            independent_returns_df = ind_returns_fn(independent_df)
            independent_returns_df["date"] = (pd.to_datetime(independent_returns_df["date"]).dt.normalize())
            
            ind_col = 'simple-returns' if self.return_type == 'simple' else 'log-returns'
            
            # Renaming the column for easier reference after merging 
            independent_returns_df = independent_returns_df.rename(columns={ind_col: 'independent_variable_returns'})
            
            independent_returns_df = independent_returns_df[['date', 'independent_variable_returns']].copy()
            
            

            self.independent_returns_data = independent_returns_df
            
            return merged_df, independent_returns_df
            
            
        elif type(self.independent) == list:
            
            independent_returns_dic = {}
            
            for asset in self.independent:
                independent = AssetData(asset, self.period, self.freq, self.start_date, self.end_date)
                        
                independent_df = independent.get_prices()
                
                ind_returns_fn = simple_returns if self.return_type == 'simple' else log_returns
                
                independent_returns_df = ind_returns_fn(independent_df)
                
                independent_returns_df["date"] = (pd.to_datetime(independent_returns_df["date"]).dt.normalize())
                
                ind_col = 'simple-returns' if self.return_type == 'simple' else 'log-returns'
                
                # # Renaming the column for easier reference after merging 
                # independent_returns_df = independent_returns_df.rename(columns={ind_col: 'independent_variable_returns'})
                
                independent_returns_df = independent_returns_df[['date', f"{ind_col}"]].copy()
                
                independent_returns_dic[asset] = independent_returns_df
                
            #self.independent_returns_dic = independent_returns_dic
                
            return merged_df, independent_returns_dic
        


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


    def _diagnostics(self):
        """This prints the results of analysis on the regression results.
        """
        
        heteroskedasticity(self.olsresults)
        
        autocorrelation(self.olsresults)
        
        normality(self.olsresults)
        
        print('\n\n\n')


# Example usage

if __name__ == '__main__':
    portfolio_dic = {
        'msft': 14.28,
        'nvda': 35.79,
        'aapl': 8.12,
        'ko': 22.45,
        'goog': 19.36
    }
    
    portfolio = PortfolioBeta(
        portfolio_dic = portfolio_dic,
        frequency = 'daily',
        asset_to_be_regressed=['spy','qqqm']
    )
    
    portfolio.summary()
    portfolio.plot_results()

    portfolio.historical_rolling_beta()
    portfolio.rolling_beta_summary()
    portfolio.rolling_beta_plot()

