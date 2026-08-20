import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from data_collection.tiingo_api import TiingoApi
from data_collection.fred_api import FredApi
from data_collection.ff_factors_api import FrenchApi
from pprint import pprint
from functools import reduce


class EquityFactorsRegression:
    """
    Tool that shows the factor betas of a particular asset
    by regressing asset's excess returns against factor excess returns

    Parameters
    ----------
    factor_source : str
        Choose between:
            "etf"
            "french" (Default)
        For the factor returns data.

        French source will use French factor return data,
        providing the proper 5 FF factors, though the latest
        data may not be available.

        ETF source will use ETF returns as proxy,
        and only provide a regression against market excess, smb, hml, mtum, rmw
        The risk free rate is calculated using the US3M t-bill

    start_date : str
        In the format (YYYY-mm-dd)
        choose when to start the observation
        if left empty, oldest available factor data will be used,
        subject to compatibility with asset returns history

    end_date : str
        In the format (YYYY-mm-dd)
        choose when to end the observation
        if left empty, newest available factor data will be used,
        subject to compatibility with asset returns history

    return_type : str
        Choose between:
            "simple" (Default)
            "log"
        Calculation of returns using simple returns or log returns
    
    frequency : str
        One of:
            "daily"
            "monthly"
            "annually"
    
    hac : int
        Choose the lag window for
        heteroscedasticity and autocorrelation robust (HAC)
        If left empty, default will be automatic

    """

    FREQ_DICT = {
        "daily": "d",
        "weekly":"w",
        "monthly":"m",
        "annually":"a"
    }

    FACTOR_SOURCE = {'french','etf'}

    RETURN_TYPE = {'simple','log'}

    MERGED_DF_COLS_SIMPLE = [
        'spy_excess_simple_return',
        'smb_simple_return',
        'hml_simple_return',
        'mtum_simple_return',
        'rmw_simple_return'
    ]

    MERGED_DF_COLS_LOG = [
        'spy_excess_log_return',
        'smb_log_return',
        'hml_log_return',
        'mtum_log_return',
        'rmw_log_return'
    ]
    FRENCH_FACTOR_COLS = [
        "Mkt-RF",
        "SMB",
        "HML",
        "RMW",
        "CMA"
    ]

    def __init__(self, factor_source: str = "french", start_date = None, end_date = None, return_type:str = 'simple', frequency:str = 'daily', hac="auto"):

        #if start date is None, API will pull from the oldest date possible of all data sources
        #if end date is None, API will pull till the latest possible date of all data sources

        if factor_source not in self.FACTOR_SOURCE:
            raise ValueError("input proper factor_source: 'french' (default) or 'etf'")

        if return_type not in self.RETURN_TYPE:
            raise ValueError("input proper return_type: 'log' or 'simple'")

        if frequency not in self.FREQ_DICT.keys():
            raise ValueError("available input frequency: 'daily', 'weekly', 'monthly', 'annually'")
      

        self.return_type = return_type
        self.factor_source = factor_source

        self.freq = frequency
        self.tiingo_freq = frequency
        self.fred_freq = self.FREQ_DICT[frequency]

        self.start = start_date
        self.end = end_date


        # hac : heteroscedasticity and autocorrelation robust (HAC) using n lags
        #
        # this is for the regression;
        #
        # users can override their hac lag number, but default will be automatic
        self.hac_lags = self._resolve_hac_lags(hac)
        self.hac_auto = True if hac == 'auto' else False


# ------------------------------
# 
#          PUBLIC APIs
#
# ------------------------------

    # =========================
    #
    #   Sets the assets for regression
    #
    # =========================
    def asset_list(self, *args:str):
        """
        Input the list of asset tickers that you want to do factor regression on.

        Example usage:
            .asset_list('nvda', 'goog', 'ko', 'aapl', 'msft')
        
        It accepts an arbitrary number of tickers.

        
        """

        self.assets = []
        for ticker in args:
            if type(ticker) != str:
                raise ValueError("asset tickers can only be in string format!")
            self.assets.append(ticker.lower())

        return

    # REGRESS
    def regress(self):
        
        # how it works:
        # step1: gathers the simple and log returns df
        # for each asset in the self.assets list,
        # then stores the pair 'ticker: df' in assets_df_dic;
        # will drop the column of whichever type of return that is not needed.
        #
        #
        # step2: gets the fama-french factor data df;
        # each column calculated based on
        # whether simple or log returns wanted
        #
        #
        # step3: merges each asset return df
        # in the assets_df_dic with the french factor data df
        # after aligning the data points for daily/monthly/annually returns,
        # then calculates the excess asset returns. 
        # then stores the asset: merged_df in merged_df_dic
        #
        #
        # step4: for each merged_df in merged_df_dic,
        # do the OLS regression of excess asset
        # against the five factors returns with statsmodel
        # with the set HAC maxlags and
        # store the model results in results dic


        if self.factor_source == 'etf':
            return self._etf_regress()
        elif self.factor_source == 'french':
            return self._french_regress()

    # =========================
    #
    #   Prints the simplified factor results
    #
    # =========================
    def results(self):
        
        if self.factor_source == 'etf':
            for asset, dic in self._grand_results['regression_results'].items():

                return_str = ''

                title = f'\n========================\nOLS regression of {asset} against a proxy basket of ETFs\n========================\n'


                window = f'{dic['model']['frequency'].capitalize()} data from {dic['model']['start_date']} to {dic['model']['end_date']} with {dic['model']['n_observations']} observations was used.\n\n\n\n'

                alpha = dic['alpha']['amount']

                alpha_t = dic['alpha']['t-stat']

                alpha_stat = 'Alpha: ' + f'{(alpha*100):.2f}% /month; t-stat: {alpha_t:5f}'
                
                if abs(alpha_t) > 2:
                    alpha_stat += '  - Statistically significant\n\n'
                else:
                    alpha_stat += '  - Statistically insignificant\n\n'

                r_square = f"R-Squared: {(dic['model']['r_squared']*100):.2f}%\nThe factors explain approximately {(dic['model']['r_squared']*100):.2f}% of {asset}'s {dic['model']['frequency']} excess-return variation over the sample.\n\n"

                

                market = dic['exposures']['market']['beta']

                market_t = dic['exposures']['market']['t-stat']

                size = dic['exposures']['size']['beta']

                size_t = dic['exposures']['size']['t-stat']

                value = dic['exposures']['value']['beta']

                value_t = dic['exposures']['value']['t-stat']

                momentum = dic['exposures']['momentum']['beta']

                momentum_t = dic['exposures']['momentum']['t-stat']

                profitability = dic['exposures']['profitability']['beta']

                profitability_t = dic['exposures']['profitability']['t-stat']

                stats = f'\nmarket beta: {market:.5f}; t-stat: {market_t:.5f}\nsize beta: {size:.5f}; t-stat: {size_t:.5f}\nvalue beta: {value:.5f}; t-stat: {value_t:.5f}\nmomentum beta: {momentum:.5f}; t-stat: {momentum_t:.5f}\nprofitability beta: {profitability:.5f}; t-stat: {profitability_t:.5f}\n\n\n'
            

                if market < 0.7:
                    market_profile = 'LOWER MARKET BETA'
                elif market < 1.2:
                    market_profile = 'MODERATE MARKET BETA'
                else:
                    market_profile = 'HIGHER MARKET BETA'

                if size < -0.2:
                    size_profile = 'LARGER CAP'
                elif size <= 0.2:
                    size_profile = 'NEUTRAL SIZE EXPOSURE'
                else:
                    size_profile = 'SMALLER CAP'

                if value < -0.2:
                    value_profile = 'GROWTH TILT'
                elif value <= 0.2:
                    value_profile = 'NEUTRAL VALUE/GROWTH'
                else:
                    value_profile = 'VALUE TILT'

                if momentum < -0.2:
                    momentum_profile = 'LOW MOMENTUM'
                elif momentum <= 0.2:
                    momentum_profile = 'NEUTRAL MOMENTUM'
                else:
                    momentum_profile = 'STRONG MOMENTUM'

                if profitability < -0.2:
                    profitability_profile = 'NEGATIVE PROFITABILITY EXPOSURE'
                elif profitability <= 0.2:
                    profitability_profile = 'NEUTRAL PROFITABILITY EXPOSURE'
                else:
                    profitability_profile = 'POSITIVE PROFITABILITY EXPOSURE'


                profile = 'FACTOR PROFILE:\n'

                profile += market_profile
                if abs(market_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                profile += size_profile
                if abs(size_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                profile += value_profile
                if abs(value_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                profile += momentum_profile
                if abs(momentum_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                profile += profitability_profile
                if abs(profitability_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                return_str += title
                return_str += window
                return_str += alpha_stat
                return_str += r_square
                return_str += stats
                return_str += profile

                print(return_str)

            print('For more information, use .advanced_results()\n')
            print('***Disclaimer: ETF proxy regression here does NOT provide the true Fama-French 5 factor regression results***\n\n')

        else:
            for asset, dic in self._grand_results['regression_results'].items():

                return_str = ''

                title = f'\n=========================================\n {asset} Fama-French 5 factor analysis\n=========================================\n'


                window = f'{dic['model']['frequency'].capitalize()} data from {dic['model']['start_date']} to {dic['model']['end_date']} with {dic['model']['n_observations']} observations was used.\n\n\n\n'

                alpha = dic['alpha']['amount']

                alpha_t = dic['alpha']['t-stat']

                alpha_stat = 'Alpha: ' + f'{(alpha*100):.2f}% /month; t-stat: {alpha_t:5f}'

                if abs(alpha_t) > 2:
                    alpha_stat += '  - Statistically significant\n\n'
                else:
                    alpha_stat += '  - Statistically insignificant\n\n'

                r_square = f"R-Squared: {(dic['model']['r_squared']*100):.2f}%\nThe factors explain approximately {(dic['model']['r_squared']*100):.2f}% of {asset}'s {dic['model']['frequency']} excess-return variation over the sample.\n\n"



                market = dic['exposures']['market']['beta']

                market_t = dic['exposures']['market']['t-stat']

                size = dic['exposures']['size']['beta']

                size_t = dic['exposures']['size']['t-stat']

                value = dic['exposures']['value']['beta']

                value_t = dic['exposures']['value']['t-stat']

                investment = dic['exposures']['investment']['beta']

                investment_t = dic['exposures']['investment']['t-stat']

                profitability = dic['exposures']['profitability']['beta']

                profitability_t = dic['exposures']['profitability']['t-stat']

                stats = f'\nmarket beta: {market:.5f}; t-stat: {market_t:.5f}\nsize beta: {size:.5f}; t-stat: {size_t:.5f}\nvalue beta: {value:.5f}; t-stat: {value_t:.5f}\nprofitability beta: {profitability:.5f}; t-stat: {profitability_t:.5f}\ninvestment beta: {investment:.5f}; t-stat: {investment_t:.5f}\n\n\n'


                if market < 0.7:
                    market_profile = 'LOWER MARKET BETA'
                elif market < 1.2:
                    market_profile = 'MODERATE MARKET BETA'
                else:
                    market_profile = 'HIGHER MARKET BETA'

                if size < -0.2:
                    size_profile = 'LARGER CAP'
                elif size <= 0.2:
                    size_profile = 'NEUTRAL SIZE EXPOSURE'
                else:
                    size_profile = 'SMALLER CAP'

                if value < -0.2:
                    value_profile = 'GROWTH TILT'
                elif value <= 0.2:
                    value_profile = 'NEUTRAL VALUE/GROWTH'
                else:
                    value_profile = 'VALUE TILT'

                if investment < -0.2:
                    investment_profile = 'MORE AGGRESSIVE INVESTMENT'
                elif investment <= 0.2:
                    investment_profile = 'NEUTRAL INVESTMENT'
                else:
                    investment_profile = 'MORE CONSERVATIVE INVESTMENT'

                if profitability < -0.2:
                    profitability_profile = 'NEGATIVE PROFITABILITY EXPOSURE'
                elif profitability <= 0.2:
                    profitability_profile = 'NEUTRAL PROFITABILITY EXPOSURE'
                else:
                    profitability_profile = 'POSITIVE PROFITABILITY EXPOSURE'


                profile = 'FACTOR PROFILE:\n'

                profile += market_profile
                if abs(market_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                profile += size_profile
                if abs(size_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                profile += value_profile
                if abs(value_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                profile += profitability_profile
                if abs(profitability_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                profile += investment_profile
                if abs(investment_t) < 2:
                    profile += '    -Not significant\n'
                else:
                    profile += '    -Significant\n'

                return_str += title
                return_str += window
                return_str += alpha_stat
                return_str += r_square
                return_str += stats
                return_str += profile

                print(return_str)

            print('For more information, use .advanced_results()\n')


    # =========================
    #
    #   Prints the advanced statistics
    #
    # =========================
    def advanced_results(self):
        for model in self._model_results_dic.values():
            print(model.summary())


    # User Help menu
    def help(self):
        print(
            """
            Usage:

            .regress() - does the regression. Important! do this first before using the below tools.

            .results() - shows the regression summary

            .advanced_results() - shows the advanced regression statistics


            """
        )

    # Removes all the assets from the list
    def reset_assets(self):
        self.assets = []

    # ----------------
    # private methods
    # ----------------

    def _etf_regress(self):

        grand_results = {}

        grand_results['basic_information'] = {
            'observations_frequency': self.freq,
            'return_convention': self.return_type,
            'factor_data_source': f"non standard etf-proxies",
            'HAC_lags': self.hac_lags,
            'asset_list': self.assets.copy()
        }
    

        asset_excess_returns_dic = {}

        for asset in self.assets:
            asset_excess_df = self._get_asset_excess_returns(asset)

            asset_excess_returns_dic[asset] = asset_excess_df.copy()

        market_fac, smb_fac, hml_fac, mtum_fac, rmw_fac = self._get_factor_returns()

        merged_df_dic = {}
        
        grand_results["merged_data"] = {}

        for asset, asset_excess_returns_df in asset_excess_returns_dic.items():
            merged_df = self._merge_factors(
                market_fac,
                smb_fac,
                hml_fac,
                mtum_fac,
                rmw_fac
            )
            

            final_merged_df = asset_excess_returns_df.merge(
                merged_df,
                on="period",
                how="inner"
            )
            
            merged_df_dic[asset] = final_merged_df
            
            

            # alert users which date rows are collapsed due to the inner merging
            asset_start = self._get_tiingo_df(asset)["date"].min()
            asset_end = self._get_tiingo_df(asset)["date"].max()
            f_start = merged_df['period'].min()
            f_end = merged_df['period'].max()           
            actual_start = final_merged_df['period'].min()
            actual_end = final_merged_df['period'].max()
            
            if self.freq == 'daily':
                asset_start_str = asset_start.strftime('%Y-%m-%d')
                asset_end_str = asset_end.strftime('%Y-%m-%d')
                f_start_str = f_start.strftime('%Y-%m-%d')
                f_end_str = f_end.strftime('%Y-%m-%d')
                actual_start_str = actual_start.strftime('%Y-%m-%d')
                actual_end_str = actual_end.strftime('%Y-%m-%d')
            
            if self.freq == 'monthly':
                asset_start_str = asset_start.strftime('%Y-%m')
                asset_end_str = asset_end.strftime('%Y-%m')
                f_start_str = f_start.strftime('%Y-%m')
                f_end_str = f_end.strftime('%Y-%m')
                actual_start_str = actual_start.strftime('%Y-%m')
                actual_end_str = actual_end.strftime('%Y-%m')
            
            if self.freq == 'annually':
                asset_start_str = asset_start.strftime('%Y')
                asset_end_str = asset_end.strftime('%Y')
                f_start_str = f_start.strftime('%Y')
                f_end_str = f_end.strftime('%Y')
                actual_start_str = actual_start.strftime('%Y')
                actual_end_str = actual_end.strftime('%Y')

            if actual_start > asset_start:
                print(f'Start date of {asset} observation window has been\n',
                        f'pushed forward from {asset_start_str} to {actual_start_str}\n',
                        f'due to data range overlap compatibility.\n'
                )
            if actual_end < asset_end:
                print(f'End date of {asset} observation window has been\n',
                        f'pushed back from {asset_end_str} to {actual_end_str}\n',
                        f'due to data range overlap compatibility.\n'
                )
            print(f'{asset} observation window is from\n',
                    f'{actual_start_str} to {actual_end_str}\n\n'
            )

            grand_results["merged_data"][asset] = {
                "dataframe": final_merged_df,
                f"{asset}_data_window": f"{asset_start_str} to {asset_end_str}",
                "factor_data_window": f"{f_start_str} to {f_end_str}",
                "regression_data_window": f"{actual_start_str} to {actual_end_str}",
            }

            
        grand_results['regression_results'] = {}

        model_results = {}

        for asset, df in merged_df_dic.items():

            actual_start = df["period"].min()
            actual_end = df["period"].max()
            
            if self.freq == 'daily':
                actual_start_str = actual_start.strftime('%Y-%m-%d')
                actual_end_str = actual_end.strftime('%Y-%m-%d')
            
            if self.freq == 'monthly':
                actual_start_str = actual_start.strftime('%Y-%m')
                actual_end_str = actual_end.strftime('%Y-%m')
            
            if self.freq == 'annually':
                actual_start_str = actual_start.strftime('%Y')
                actual_end_str = actual_end.strftime('%Y')


            y_col = f'{asset}_excess_simple_return' if self.return_type == 'simple' else f'{asset}_excess_log_return'
            x_col = self.MERGED_DF_COLS_SIMPLE if self.return_type == 'simple' else self.MERGED_DF_COLS_LOG

            y = df[y_col] 
            X = df[x_col]

            model = self._ols_regression(y, X)

            model_results[asset] = model

            grand_results['regression_results'][asset] = {
                "summary": {
                    "name": "ETF Proxy Factor Regression",
                    "type": "OLS",
                    "dependent_variable": f"{asset}-excess-returns",
                    "factors": x_col,
                },
                "model": {
                    "frequency": f"{self.freq}",
                    "start_date": f"{actual_start_str}",
                    "end_date": f"{actual_end_str}",
                    "n_observations": int(model.nobs),
                    "r_squared": model.rsquared.item(),
                    "adjusted_r_squared": model.rsquared_adj.item()
                },
                "specification": {
                    "dependent_variable": f"{asset}-spy",
                    "independent_variables": ["SPY excess return", "ETF size proxy", "ETF value/growth proxy", "momentum ETF proxy","profitability ETF proxy"],
                    "intercept": True,
                },
                "inference": {
                    "covariance_type": f'{"HAC" if self.hac_lags > 0 else 'Standard'}',
                    "hac_maxlags": f'{self.hac_lags}',
                    "hac_selection": f'{'frequency_default' if self.hac_auto else 'user_defined'}'
                },
                "alpha":{
                    "amount": model.params['const'].item(),
                    "standard_error": model.bse['const'].item(),
                    "t-stat": model.tvalues['const'].item(),
                    "p-value":model.pvalues['const'].item(),
                    "ci_lower": model.conf_int().loc['const', 0].item(),
                    "ci_upper": model.conf_int().loc['const', 1].item(),
                },
                "exposures":{
                    "market":{
                        "beta": model.params[x_col[0]].item(),
                        "standard_error": model.bse[x_col[0]].item(),
                        "t-stat": model.tvalues[x_col[0]].item(),
                        "p-value": model.pvalues[x_col[0]].item(),
                        "ci_lower": model.conf_int().loc[x_col[0], 0].item(),
                        "ci_upper": model.conf_int().loc[x_col[0], 1].item(),
                    },
                    "size":{
                        "beta": model.params[x_col[1]].item(),
                        "standard_error": model.bse[x_col[1]].item(),
                        "t-stat": model.tvalues[x_col[1]].item(),
                        "p-value": model.pvalues[x_col[1]].item(),
                        "ci_lower": model.conf_int().loc[x_col[1], 0].item(),
                        "ci_upper": model.conf_int().loc[x_col[1], 1].item(),
                    },
                    "value":{
                        "beta": model.params[x_col[2]].item(),
                        "standard_error": model.bse[x_col[2]].item(),
                        "t-stat": model.tvalues[x_col[2]].item(),
                        "p-value": model.pvalues[x_col[2]].item(),
                        "ci_lower": model.conf_int().loc[x_col[2], 0].item(),
                        "ci_upper": model.conf_int().loc[x_col[2], 1].item(),
                    },
                    "momentum":{
                        "beta": model.params[x_col[3]].item(),
                        "standard_error": model.bse[x_col[3]].item(),
                        "t-stat": model.tvalues[x_col[3]].item(),
                        "p-value": model.pvalues[x_col[3]].item(),
                        "ci_lower": model.conf_int().loc[x_col[3], 0].item(),
                        "ci_upper": model.conf_int().loc[x_col[3], 1].item(),
                    },
                    "profitability":{
                        "beta": model.params[x_col[4]].item(),
                        "standard_error": model.bse[x_col[4]].item(),
                        "t-stat": model.tvalues[x_col[4]].item(),
                        "p-value": model.pvalues[x_col[4]].item(),
                        "ci_lower": model.conf_int().loc[x_col[4], 0].item(),
                        "ci_upper": model.conf_int().loc[x_col[4], 1].item(),
                    }
                }
            }


        self._model_results_dic = model_results
        self._grand_results = grand_results
                
        # for model in model_results.values():
        #     print(model.summary())

        # return grand_results

    def _french_regress(self):
        if self.freq not in {'daily','monthly','annually'}:
            raise ValueError("using french factor data is only available in input frequency: 'daily', OR 'monthly' OR 'annually'; otherwise use 'etf' as factor_source")

        grand_results = {}

        grand_results['basic_information'] = {
            'observations_frequency': self.freq,
            'return_convention': self.return_type,
            'factor_data_source': self.factor_source,
            'HAC_lags': self.hac_lags,
            'asset_list': self.assets.copy()
        }
        

        if self.freq == 'daily':
            french_factor_df = self._load_french('daily').copy()
        elif self.freq == 'monthly':
            french_factor_df = self._load_french('monthly').copy()
        else:
            french_factor_df = self._load_french('annually').copy()

        # grand_results['french_data'] = french_factor_df
        # grand_results['assets_data'] = {}


        assets_df_dic = {}

        for ticker in self.assets:
            asset_returns_df = self._get_tiingo_df(ticker).copy()

            if not asset_returns_df.empty:
                assets_df_dic[ticker] = asset_returns_df

                # grand_results['assets_data'][ticker] = asset_returns_df


        assets_returns_df_dic = {}


        if self.return_type == 'log':
            # the french factor date is calculated with simple returns
            # convert to log returns if needed
            french_factor_df["Mkt"] = french_factor_df["Mkt-RF"] + french_factor_df["RF"]

            french_factor_df["log_Mkt"] = np.log1p(french_factor_df["Mkt"])
            french_factor_df["RF"] = np.log1p(french_factor_df["RF"])

            french_factor_df["log_Mkt_excess"] = (
                french_factor_df["log_Mkt"] - french_factor_df["RF"]
            )

            french_factor_df = french_factor_df.drop(columns=['log_Mkt','Mkt','Mkt-RF'])

            french_factor_df.rename(columns={'log_Mkt_excess': 'Mkt-RF'}, inplace=True)

            cols = ['SMB', 'HML', 'RMW', 'CMA']
            french_factor_df[cols] = np.log1p(french_factor_df[cols])


            

            for ticker, df in assets_df_dic.items():
                assets_returns_df_dic[ticker] = df[['date', f'{ticker}_log_returns']].copy()


        else:
            for ticker, df in assets_df_dic.items():
                assets_returns_df_dic[ticker] = df[['date', f'{ticker}_simple_returns']].copy()

        
        grand_results['merged_data'] = {}




        if self.freq == 'daily':
                        
            merged_df_dic = {}
            
            for ticker, df in assets_returns_df_dic.items():

                merged_df = df.copy()

                merged_df = merged_df.merge(
                    french_factor_df,
                    on="date",
                    how="inner",
                    validate="one_to_one"
                )

                merged_df_dic[ticker] = merged_df

                
                # alert users which date rows are collapsed due to the inner merging
                asset_start = df["date"].min()
                asset_end = df["date"].max()

                ff_start = french_factor_df["date"].min()
                ff_end = french_factor_df["date"].max()
                
                actual_start = merged_df['date'].min()
                actual_end = merged_df['date'].max()

                if actual_start > asset_start:
                    print(f'Start date of {ticker} observation window has been\n',
                            f'pushed forward from {asset_start.strftime('%Y-%m-%d')} to {actual_start.strftime('%Y-%m-%d')}\n',
                            f'due to data range overlap compatibility.\n'
                    )
                if actual_end < asset_end:
                    print(f'End date of {ticker} observation window has been\n',
                            f'pushed back from {asset_end.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}\n',
                            f'due to data range overlap compatibility.\n'
                    )
                print(f'{ticker} observation window is from\n',
                        f'{actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}\n\n'
                )

                grand_results['merged_data'][ticker] = {
                    "dataframe": merged_df,
                    f'{ticker}_data_window': f'{asset_start.strftime('%Y-%m-%d')} to {asset_end.strftime('%Y-%m-%d')}',
                    f'factor_data_window': f'{ff_start.strftime('%Y-%m-%d')} to {ff_end.strftime('%Y-%m-%d')}',
                    f'regression_data_window': f'{actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}'
                }

            
        elif self.freq == 'monthly':

            merged_df_dic = {}

            french_factor_df = french_factor_df.copy()

            french_factor_df["__period"] = (french_factor_df["date"].dt.to_period("M"))


            for ticker, df in assets_returns_df_dic.items():

                merged_df = df.copy()

                merged_df["__period"] = merged_df["date"].dt.to_period("M")
                
                merged_df = merged_df.merge(
                    french_factor_df,
                    on="__period",
                    how="inner",
                    validate="one_to_one",
                    suffixes=("", "_ff")
                )

                merged_df = merged_df.drop(columns = ['__period','date_ff'])

                merged_df_dic[ticker] = merged_df

                # alert users which date rows are collapsed due to the inner merging
                asset_start = df["date"].min()
                asset_end = df["date"].max()

                ff_start = french_factor_df["date"].min()
                ff_end = french_factor_df["date"].max()
            
                actual_start = merged_df['date'].min()
                actual_end = merged_df['date'].max()

                if actual_start > asset_start:
                    print(f'Start date of {ticker} observation window has been\n',
                            f'pushed forward from {asset_start.strftime('%Y-%m-%d')} to {actual_start.strftime('%Y-%m-%d')}\n',
                            f'due to data range overlap compatibility.\n'
                    )
                if actual_end < asset_end:
                    print(f'End date of {ticker} observation window has been\n',
                            f'pushed back from {asset_end.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}\n',
                            f'due to data range overlap compatibility.\n'
                    )
                print(f'{ticker} observation window is from\n',
                        f'{actual_start.strftime('%Y-%m')} to {actual_end.strftime('%Y-%m')}'
                )

                grand_results['merged_data'][ticker] = {
                    "dataframe": merged_df,
                    f'{ticker}_data_window': f'{asset_start.strftime('%Y-%m')} to {asset_end.strftime('%Y-%m')}',
                    f'factor_data_window': f'{ff_start.strftime('%Y-%m')} to {ff_end.strftime('%Y-%m')}',
                    f'regression_data_window': f'{actual_start.strftime('%Y-%m')} to {actual_end.strftime('%Y-%m')}'
                }
            
            
        elif self.freq == 'annually':

            merged_df_dic = {}

            french_factor_df = french_factor_df.copy()

            french_factor_df["__period"] = (french_factor_df["date"].dt.year)


            for ticker, df in assets_returns_df_dic.items():

                merged_df = df.copy()

                merged_df["__period"] = merged_df["date"].dt.year
                
                merged_df = merged_df.merge(
                    french_factor_df,
                    on="__period",
                    how="inner",
                    validate="one_to_one",
                    suffixes=("", "_ff")
                )

                merged_df = merged_df.drop(columns = ['__period','date_ff'])

                merged_df_dic[ticker] = merged_df


                asset_start = df["date"].min()
                asset_end = df["date"].max()

                ff_start = french_factor_df["date"].min()
                ff_end = french_factor_df["date"].max()

                actual_start = merged_df['date'].min()
                actual_end = merged_df['date'].max()
                
                if actual_start > asset_start:
                    print(f'Start date of {ticker} observation window has been\n',
                            f'pushed forward from {asset_start.strftime('%Y-%m-%d')} to {actual_start.strftime('%Y-%m-%d')}\n',
                            f'due to data range overlap compatibility.\n'
                    )
                if actual_end < asset_end:
                    print(f'End date of {ticker} observation window has been\n',
                            f'pushed back from {asset_end.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}\n',
                            f'due to data range overlap compatibility.\n'
                    )
                    print(f'{ticker} observation window is from\n',
                            f'{actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}'
                    )

                grand_results['merged_data'][ticker] = {
                    "dataframe": merged_df,
                    f'{ticker}_data_window': f'{asset_start.strftime('%Y')} to {asset_end.strftime('%Y')}',
                    f'factor_data_window': f'{ff_start.strftime('%Y')} to {ff_end.strftime('%Y')}',
                    f'regression_data_window': f'{actual_start.strftime('%Y')} to {actual_end.strftime('%Y')}'
                }



        #   creating a column for asset excess returns
        if self.return_type == "simple":

            for ticker, df in merged_df_dic.items():

                df[f'{ticker}-RF'] = df[f'{ticker}_simple_returns'] - df['RF']
        else:

            for ticker, df in merged_df_dic.items():

                df[f'{ticker}-RF'] = df[f'{ticker}_log_returns'] - df['RF']



        
        grand_results['regression_results'] = {}

        model_results = {}

        for ticker, df in merged_df_dic.items():

            actual_start = df["date"].min()
            actual_end = df["date"].max()
            
            if self.freq == 'daily':
                actual_start_str = actual_start.strftime('%Y-%m-%d')
                actual_end_str = actual_end.strftime('%Y-%m-%d')
            
            if self.freq == 'monthly':
                actual_start_str = actual_start.strftime('%Y-%m')
                actual_end_str = actual_end.strftime('%Y-%m')
            
            if self.freq == 'annually':
                actual_start_str = actual_start.strftime('%Y')
                actual_end_str = actual_end.strftime('%Y')

            y = df[f'{ticker}-RF']
            X = df[self.FRENCH_FACTOR_COLS]

            model = self._ols_regression(y, X)

            model_results[ticker] = model


            grand_results['regression_results'][ticker] = {
                "summary": {
                    "name": "Fama-French 5 Factor Regression",
                    "type": "OLS",
                    "dependent_variable": f"{ticker}-RF",
                    "factors": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
                },
                "model": {
                    "frequency": f"{self.freq}",
                    "start_date": f"{actual_start_str}",
                    "end_date": f"{actual_end_str}",
                    "n_observations": int(model.nobs),
                    "r_squared": model.rsquared.item(),
                    "adjusted_r_squared": model.rsquared_adj.item()
                },
                "specification": {
                    "dependent_variable": f"{ticker}-RF",
                    "independent_variables": ["Mkt-RF", "SMB", "HML", "RMW","CMA"],
                    "intercept": True,
                },
                "inference": {
                    "covariance_type": f'{"HAC" if self.hac_lags > 0 else 'Standard'}',
                    "hac_maxlags": f'{self.hac_lags}',
                    "hac_selection": f'{'frequency_default' if self.hac_auto else 'user_defined'}'
                },
                "alpha":{
                    "amount": model.params['const'].item(),
                    "standard_error": model.bse['const'].item(),
                    "t-stat": model.tvalues['const'].item(),
                    "p-value":model.pvalues['const'].item(),
                    "ci_lower": model.conf_int().loc['const', 0].item(),
                    "ci_upper": model.conf_int().loc['const', 1].item(),
                },
                "exposures":{
                    "market":{
                        "beta": model.params['Mkt-RF'].item(),
                        "standard_error": model.bse['Mkt-RF'].item(),
                        "t-stat": model.tvalues['Mkt-RF'].item(),
                        "p-value": model.pvalues['Mkt-RF'].item(),
                        "ci_lower": model.conf_int().loc['Mkt-RF', 0].item(),
                        "ci_upper": model.conf_int().loc['Mkt-RF', 1].item(),
                    },
                    "size":{
                        "beta": model.params["SMB"].item(),
                        "standard_error": model.bse["SMB"].item(),
                        "t-stat": model.tvalues["SMB"].item(),
                        "p-value": model.pvalues["SMB"].item(),
                        "ci_lower": model.conf_int().loc["SMB", 0].item(),
                        "ci_upper": model.conf_int().loc["SMB", 1].item(),
                    },
                    "value":{
                        "beta": model.params["HML"].item(),
                        "standard_error": model.bse["HML"].item(),
                        "t-stat": model.tvalues["HML"].item(),
                        "p-value": model.pvalues["HML"].item(),
                        "ci_lower": model.conf_int().loc["HML", 0].item(),
                        "ci_upper": model.conf_int().loc["HML", 1].item(),
                    },
                    "profitability":{
                        "beta": model.params["RMW"].item(),
                        "standard_error": model.bse["RMW"].item(),
                        "t-stat": model.tvalues["RMW"].item(),
                        "p-value": model.pvalues["RMW"].item(),
                        "ci_lower": model.conf_int().loc["RMW", 0].item(),
                        "ci_upper": model.conf_int().loc["RMW", 1].item(),
                    },
                    "investment":{
                        "beta": model.params["CMA"].item(),
                        "standard_error": model.bse["CMA"].item(),
                        "t-stat": model.tvalues["CMA"].item(),
                        "p-value": model.pvalues["CMA"].item(),
                        "ci_lower": model.conf_int().loc["CMA", 0].item(),
                        "ci_upper": model.conf_int().loc["CMA", 1].item(),
                    }
                }
            }

        self._model_results_dic = model_results
        self._grand_results = grand_results
        
        # for model in model_results.values():
        #     print(model.summary())

        # return grand_results

    def _ols_regression(self, y, X):
        X = sm.add_constant(X)

        if self.hac_lags is None:
            model = sm.OLS(y, X, missing="drop").fit()
        else:
            model = sm.OLS(y, X, missing="drop").fit(
                cov_type="HAC",
                cov_kwds={"maxlags": self.hac_lags}
            )

        return model
        
    def _get_tiingo_df(self, ticker):

        tiingo_api_key = os.getenv('TIINGO_API_KEY')

        tiingo_obj = TiingoApi(tiingo_api_key, self.tiingo_freq, True)

        df = tiingo_obj.get_data(ticker, self.start, self.end).copy()
        

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "period",
                    f"{ticker}_log_returns",
                    f"{ticker}_simple_returns"
                ]
            )
        
        df["date"] = pd.to_datetime(df["date"])

        df[f'{ticker}_simple_returns'] = (df['adjClose'] / df['adjClose'].shift(1)) - 1

        df[f'{ticker}_log_returns'] = np.log(df['adjClose'] / df['adjClose'].shift(1))

        df = df[['date', f'{ticker}_log_returns', f'{ticker}_simple_returns']]

        df.dropna(ignore_index=True, inplace=True)
        
        
        if self.freq == "daily":
            df["period"] = df["date"]

        elif self.freq == "monthly":
            df["period"] = df["date"].dt.to_period("M")

        elif self.freq == "annually":
            df["period"] = df["date"].dt.to_period("Y")

        else:
            raise ValueError(
                f"Unsupported frequency: {self.freq}"
            )

        return df[
            [
                "date",
                "period",
                f"{ticker}_log_returns",
                f"{ticker}_simple_returns"
            ]
        ]
    

    def _get_rf_df(self):
        # THIS IS FOR GETTING RF RATE FROM FRED API 3MONTH TREASURY PROXY]
        #
        # NOT USING FRENCH RF
                
        rf_ticker = "dgs3mo"

        fred_api_key = os.getenv('FRED_API_KEY')

        fred_obj = FredApi(fred_api_key, 'd')

        rf_df = fred_obj.get_data(rf_ticker,self.start, self.end).copy()
        
        
        if rf_df.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "period",
                    "rf_log_returns",
                    "rf_simple_returns"
                ]
            )
            
        rf_df["date"] = pd.to_datetime(rf_df["date"])


        # You have three calendar days between observations. If you're calculating the risk-free return from
        # Friday close → Monday close, you should account for 3 calendar days, not just one trading day.
        # 
        #
        # previous day's annualized yield
        #         ↓
        #    divide by 100
        #         ↓
        #    annual decimal rate
        #         ↓
        #   × number of days
        #         ↓
        #       / 365
        #         ↓
        #  period risk-free return
        #
        
        rf_df["days"] = rf_df["date"].diff().dt.days
        
        rf_df["rf_simple_returns"] = (
            rf_df["value"].shift(1) / 100
            * rf_df["days"]
            / 365
        )
        rf_df["rf_log_returns"] = np.log1p(rf_df["rf_simple_returns"])
        
        
        rf_df.dropna(
            subset=["rf_simple_returns", "rf_log_returns"],
            inplace=True
        )
        
        
        if self.freq == "daily":

            # Each observation already represents the RF return
            # ending on this date.
            rf_df["period"] = rf_df["date"]

            rf_df = rf_df[
                [
                    "date",
                    "period",
                    "rf_log_returns",
                    "rf_simple_returns"
                ]
            ]

        elif self.freq == "monthly":

            # Group by calendar month.
            rf_df["period"] = rf_df["date"].dt.to_period("M")

            # Log returns are additive, so sum them within the month
            # and convert back to a simple return.
            rf_monthly = (
                rf_df
                .groupby("period")["rf_log_returns"]
                .sum()
                .pipe(np.expm1)
                .reset_index(name="rf_simple_returns")
            )

            # Keep the corresponding monthly log return as well.
            rf_monthly["rf_log_returns"] = np.log1p(
                rf_monthly["rf_simple_returns"]
            )

            rf_df = rf_monthly[
                [
                    "period",
                    "rf_log_returns",
                    "rf_simple_returns"
                ]
            ]

        elif self.freq == "annually":

            # Group by calendar year.
            rf_df["period"] = rf_df["date"].dt.to_period("Y")

            # Compound the daily/interval RF returns over the year.
            rf_annual = (
                rf_df
                .groupby("period")["rf_log_returns"]
                .sum()
                .pipe(np.expm1)
                .reset_index(name="rf_simple_returns")
            )

            rf_annual["rf_log_returns"] = np.log1p(
                rf_annual["rf_simple_returns"]
            )

            rf_df = rf_annual[
                [
                    "period",
                    "rf_log_returns",
                    "rf_simple_returns"
                ]
            ]

        else:
            raise ValueError(
                f"Unsupported frequency: {self.freq}"
            )

        return rf_df
        
    def _get_asset_excess_returns(self, asset):
        
        # THIS IS FOR GETTING ASSET EXCESS FOR TIINGO ASSET DF MINUS FRED 3MONTH TREASURY PROXY]
        #
        # NOT USING FRENCH RF
        
        asset_df = self._get_tiingo_df(asset)

        rf_df = self._get_rf_df()
        
        # Both DataFrames now have a common "period" column.
        #
        # Daily:
        #   2025-02-28 == 2025-02-28
        #
        # Monthly:
        #   2025-02-28 -> 2025-02
        #   2025-02-01 -> 2025-02
        #
        # Annual:
        #   2025-12-31 -> 2025
        #   2025-01-01 -> 2025
        #
        # Therefore we don't need the actual dates to match.
        
        excess_df = asset_df.merge(
            rf_df[
                [
                    "period",
                    "rf_log_returns",
                    "rf_simple_returns"
                ]
            ],
            on="period",
            how="inner"
        )

        
        
        # Simple excess return:
        #
        # R_i,t - R_f,t
        excess_df[f"{asset}_excess_simple_return"] = (
            excess_df[f"{asset}_simple_returns"]
            - excess_df["rf_simple_returns"]
        )

        # Log excess return:
        #
        # log(1 + R_i,t) - log(1 + R_f,t)
        excess_df[f"{asset}_excess_log_return"] = (
            excess_df[f"{asset}_log_returns"]
            - excess_df["rf_log_returns"]
        )

        excess_df = excess_df[
            [
                "period",
                f"{asset}_excess_simple_return",
                f"{asset}_excess_log_return"
            ]
        ]

        excess_df.dropna(
            ignore_index=True,
            inplace=True
        )

        return excess_df  

    def _get_factor_returns(self):

        # THIS METHOD IS FOR GETTING FACTOR RETURNS THROUGH ETF PROXY
        #
        # NOT THROUGH FRENCH DATA
        #
        
        
        # market excess
        market_df = self._get_tiingo_df("spy")

        rf_df = self._get_rf_df()
        
        market_excess = market_df.merge(
            rf_df[
                [
                    "period",
                    "rf_log_returns",
                    "rf_simple_returns"
                ]
            ],
            on="period",
            how="inner"
        )
        
        market_excess[f"spy_excess_simple_return"] = (
            market_excess['spy_simple_returns'] - market_excess["rf_simple_returns"]
        )
        
        market_excess[f"spy_excess_log_return"] = (
            market_excess[f'spy_log_returns'] - market_excess["rf_log_returns"]
        )
        
        market_excess = market_excess[['period',f'spy_excess_simple_return',f'spy_excess_log_return']]
        
        market_excess = market_excess.dropna().reset_index(drop=True)
        

        #smb
        small_df = self._get_tiingo_df("iwm")
        large_df = self._get_tiingo_df("spy")
        
        smb = small_df.merge(
            large_df[
                [
                    "period",
                    "spy_log_returns",
                    "spy_simple_returns"
                ]
            ],
            on="period",
            how="inner"
        )
        
        smb['smb_simple_return'] = (
            smb['iwm_simple_returns'] - smb['spy_simple_returns']
        )

        smb['smb_log_return'] = (
            smb['iwm_log_returns'] - smb['spy_log_returns']
        )
        
        smb = smb[['period','smb_simple_return','smb_log_return']]
        smb = smb.dropna().reset_index(drop=True)


        #hml
        value_df = self._get_tiingo_df("iwd")
        growth_df = self._get_tiingo_df("iwf")
        
        hml = value_df.merge(
            growth_df[
                [
                    "period",
                    "iwf_log_returns",
                    "iwf_simple_returns"
                ]
            ],
            on="period",
            how="inner"
        )

        hml['hml_simple_return'] = (
            hml['iwd_simple_returns'] - hml['iwf_simple_returns']
        )
        hml['hml_log_return'] = (
            hml['iwd_log_returns'] - hml['iwf_log_returns']
        )
        hml = hml[['period','hml_simple_return','hml_log_return']]
        hml = hml.dropna().reset_index(drop=True)


        #momentum
        mtum_df = self._get_tiingo_df("mtum")
        market_df = self._get_tiingo_df("spy")
        
        mtum = mtum_df.merge(
            market_df[
                [
                    "period",
                    "spy_log_returns",
                    "spy_simple_returns"
                ]
            ],
            on="period",
            how="inner"
        )
        
        mtum['mtum_simple_return'] = (
            mtum['mtum_simple_returns'] - mtum['spy_simple_returns']
        )
        mtum['mtum_log_return'] = (
            mtum['mtum_log_returns'] - mtum['spy_log_returns']
        )
        mtum = mtum[['period','mtum_simple_return','mtum_log_return']]
        mtum = mtum.dropna().reset_index(drop=True)

        #RMW
        robust_df = self._get_tiingo_df("qual")
        market_df = self._get_tiingo_df("spy")
        
        rmw = robust_df.merge(
            market_df[
                [
                    "period",
                    "spy_log_returns",
                    "spy_simple_returns"
                ]
            ],
            on="period",
            how="inner"
        )
        
        rmw['rmw_simple_return'] = (
            rmw['qual_simple_returns'] - rmw['spy_simple_returns']
        )
        rmw['rmw_log_return'] = (
            rmw['qual_log_returns'] - rmw['spy_log_returns']
        )
        rmw = rmw[['period','rmw_simple_return','rmw_log_return']]
        rmw = rmw.dropna().reset_index(drop=True)
        
        return market_excess, smb, hml, mtum, rmw
        
    def _merge_factors(self, market_df, smb_df, hml_df, mtum_df, rmw_df):
        if self.return_type == 'simple':
            #assetdf = asset_excess_return_df[['date', f'{asset_name}_excess_simple_return']]
            marketdf = market_df[['period', 'spy_excess_simple_return']]
            smbdf = smb_df[['period', 'smb_simple_return']]
            hmldf = hml_df[['period', 'hml_simple_return']]
            mtumdf = mtum_df[['period', 'mtum_simple_return']]
            rmwdf = rmw_df[['period','rmw_simple_return']]
        elif self.return_type == 'log':
            #assetdf = asset_excess_return_df[['date', f'{asset_name}_excess_log_return']]
            marketdf = market_df[['period', 'spy_excess_log_return']]
            smbdf = smb_df[['period', 'smb_log_return']]
            hmldf = hml_df[['period', 'hml_log_return']]
            mtumdf = mtum_df[['period', 'mtum_log_return']]
            rmwdf = rmw_df[['period','rmw_log_return']]
        
        dfs = [marketdf, smbdf, hmldf, mtumdf, rmwdf]

        # Merge them sequentially on the 'date' column using an inner join
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='period', how='inner'), dfs)
        
        merged_df.dropna(ignore_index=True, inplace=True)
        
        return merged_df
    
    def _load_french(self, frequency):

        french_obj = FrenchApi('US', frequency)
        df = french_obj.get_data().copy()
        return df

        # WARNING: THE default dataset here is US factor data
        #
        # only use US factor data for regressing US equities
        #
        # for OTHER regions, get factor data for other regions (important!) (coming soon!)
        #
        # 

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

    def _merge(self, df1, df2):
        df = df1.merge(
            df2,
            on="date",
            how="inner"
        )

        return df


# -----------------------
# EXAMPLE USAGE
# -----------------------

if __name__ == "__main__":    
    fac1 = EquityFactorsRegression(factor_source="french", frequency='monthly')
    fac1.asset_list('goog','ko')
    fac1.reset_assets()
    fac1.asset_list('nvda','msft')
    data = fac1.regress()
    fac1.results()
    #fac1.advanced_results()
