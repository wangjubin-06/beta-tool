import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
from functools import reduce
from regression_beta.data import AssetData
from regression_beta.diagnostics import *
from regression_beta.plotting import returns_distribution_plot, two_asset_ols_plot
from regression_beta.returns import log_returns, simple_returns


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
    


    def __init__(self, portfolio_dic: dict, asset_to_be_regressed = "spy", frequency = 'daily', period = '10y', start_date = None, end_date = None, return_type = 'simple'):
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


        self.portfolio_dic = cleaned_portfolio_dic
        
        self.period = period
        
        self.freq = frequency
        
        self.start_date = start_date
        
        self.end_date = end_date
        
        self.return_type = return_type
        
        self.independent = asset_to_be_regressed.lower()
        
        
        portfolio_desc = ""
        
        for ticker, weight in self.portfolio_dic.items():
            portfolio_desc += f'{ticker} - '
            portfolio_desc += f'{weight:.4f} %  '
        
        self.portfolio_desc = portfolio_desc
        
        
        # Getting relevant returns data
        portfolio_merged_df, independent_df = self._get_data()
        
        # Regression
        self._regress(portfolio_merged_df, independent_df)


    # Public APIs
    def summary(self):
        """Return a formatted summary of the regression results."""
        
        portfolio_desc = ""
        
        for ticker, weight in self.portfolio_dic.items():
            portfolio_desc += f'{ticker}-'
            portfolio_desc += f'{weight:.4f} %\n'
        
        
        
        lines = [
            '\n\n=========================================',
            f"OLS Regression:\nPortfolio consisting of\n{portfolio_desc}against {self.independent}:",
            '=========================================',
            f"{'Beta':<30}: {self.beta:.5f}",
            f"{'Alpha':<30}: {self.intercept:.5f}",
            f"{'R-squared':<30}: {self.rsquare:.5f}",
            f"{'Alpha p-value':<30}: {self.alpha_p_value:.5f}",
            f"{'Beta p-value':<30}: {self.beta_p_value:.4g}",
            f"{'Beta t-stat':<30}: {self.beta_tstat:.5f}",
            f"{'Beta std-error':<30}: {self.beta_std_error:.5f}",
            f"{'Beta Confidence Interval high':<30}: {self.beta_ci_high:.5f}",
            f"{'Beta Confidence Interval low':<30}: {self.beta_ci_low:.5f}",
            f"{'Start date':<30}: {self.regstart}",
            f"{'End date':<30}: {self.regend}",
            f"{'Frequency':<30}: {self.freq}",
            f"{'Return type':<30}: {self.return_type}",
            f"{'No. of observations':<30}: {self.observations}",
            f"{'Residual volatility':<30}: {self.residual_vol:.5f}",
        ]
        print("\n".join(lines))
        
        self._diagnostics()


    def plot_results(self):
        fig = plt.figure(
            figsize=(14,10),
            #layout="constrained"
        )

        gs = fig.add_gridspec(
            nrows=2,
            ncols=4,
            height_ratios=[1.0, 1.5],
        )

        # ax_asset_1_price = fig.add_subplot(gs[0, :2])
        # ax_asset_2_price = fig.add_subplot(gs[0, 2:4])
        ax_asset_1_dist = fig.add_subplot(gs[0, 0])
        ax_asset_1_qq = fig.add_subplot(gs[0, 1])
        ax_asset_2_dist = fig.add_subplot(gs[0, 2])
        ax_asset_2_qq = fig.add_subplot(gs[0, 3])
        
        ax_ols = fig.add_subplot(gs[1, :])

        # plotting.price_series_plot(ticker = self.asset1, data=self.asset_1_prices, ax = ax_asset_1_price)
        # plotting.price_series_plot(ticker = self.asset2, data=self.asset_2_prices, ax = ax_asset_2_price)


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


    def historical_rolling_beta(self, window=60):
        pass


    # Private methods
    def _regress(self, merged_df, independent_df):
        
        df = pd.merge(merged_df, independent_df, on="date", how="inner")
        
        df = df.dropna()
        
        # Alert user of the final dates
        
        start = df['date'].min()
        end = df['date'].max()
        
        self.regstart = start
        self.regend = end
        
        print(f"Regression date is from {start} to {end} so that historical data overlap across all assets involved.")
        
        if self.return_type == 'log':
            y = df['portfolio_log_returns']
        else:
            y = df['portfolio_simple_returns']
        x = df['independent_variable_returns']
        
        x = sm.add_constant(x)

        # Fit the OLS model
        model = sm.OLS(y, x).fit()
        
        results = model
        
        self.olsresults = model
        
        stats = {
            "beta": results.params['independent_variable_returns'],
            "intercept": results.params['const'],
            "r_squared": results.rsquared,
            "alpha_p_value": results.pvalues['const'],
            "beta_pvalue": results.pvalues['independent_variable_returns'],
            "beta_tstat": results.tvalues['independent_variable_returns'],
            "beta_std_error": results.bse[f'independent_variable_returns'],
            "beta_ci_low": results.conf_int().loc['independent_variable_returns', 0],
            "beta_ci_high": results.conf_int().loc['independent_variable_returns', 1],
            "start_date": self.regstart,
            "end_date": self.regend,
            "n_obs": results.nobs,
            "residual_vol": results.resid.std()
        }

        self.beta = float(stats["beta"])
        self.intercept = float(stats["intercept"])
        self.rsquare = float(stats["r_squared"])
        self.alpha_p_value = float(stats['alpha_p_value'])
        self.beta_p_value = float(stats["beta_pvalue"])
        self.beta_tstat = float(stats["beta_tstat"])
        self.beta_std_error = float(stats["beta_std_error"])
        self.beta_ci_low = float(stats["beta_ci_low"])
        self.beta_ci_high = float(stats["beta_ci_high"])
        self.observations = int(stats["n_obs"])
        self.residual_vol = float(stats["residual_vol"])



    def _get_data(self):
        
        # Getting price data for each asset
        prices_dict = {}
        
        for asset in self.portfolio_dic.keys():
            
            dataobj = AssetData(asset, self.period, self.freq, self.start_date, self.end_date)
            
            df = dataobj.get_prices()
            
            if df.empty:
                raise ValueError(f"No price data available for {asset}!")
            
            prices_dict[asset.lower()] = df
        
        
        # Getting weighted returns data for each asset
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
        
        
        
        # Get returns data for independent variable
        independent = AssetData(self.independent, self.period, self.freq, self.start_date, self.end_date)
                    
        independent_df = independent.get_prices()
        
        ind_returns_fn = simple_returns if self.return_type == 'simple' else log_returns
        
        independent_returns_df = ind_returns_fn(independent_df)
        independent_returns_df["date"] = (pd.to_datetime(independent_returns_df["date"]).dt.normalize())
        
        ind_col = 'simple-returns' if self.return_type == 'simple' else 'log-returns'
        
        # Renaming the column for easier reference after merging 
        independent_returns_df = independent_returns_df.rename(columns={ind_col: 'independent_variable_returns'})
        
        independent_returns_df = independent_returns_df[['date', 'independent_variable_returns']].copy()
        
        
        self.portfolio_returns_data = merged_df
        self.independent_returns_data = independent_returns_df
        
        
        return merged_df, independent_returns_df



    def _diagnostics(self):
        """This prints the results of analysis on the regression results.
        """
        
        heteroskedasticity(self.olsresults)
        
        autocorrelation(self.olsresults)
        
        normality(self.olsresults)
        
        print('\n\n\n')


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
        frequency = 'daily'
    )
    
    portfolio.summary()
    portfolio.plot_results()

