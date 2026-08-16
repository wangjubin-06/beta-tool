import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
# from data import AssetData
# from beta_tool.regression_beta.returns import log_returns, simple_returns
# from beta_tool.factor_research.riskfreerate import generate_risk_free_rate_df
from datetime import date

import os
from data_collection.tiingo_api import TiingoApi
from data_collection.fred_api import FredApi
from data_collection.ff_factors_api import FrenchApi
from regression_beta.returns import simple_returns, log_returns
import requests
from io import BytesIO, StringIO
import zipfile
from pprint import pprint


class EquityFactorsRegression():
    """
    
    this class produces an object that shows the factor betas of a particular asset
    
    the betas comes from regressing the asset's equity premium (asset return - risk free rate) against market (rm - rf) returns, SMB (small-caps) returns, HML (value) returns, MOM (momentum) returns, RMW (quality)
    
    
    the returns for each factor are proxied using ETFs, instead of actual Fama-French factor returns for simplicity. a new version with the actual returns may come.

    RMW (quality) returns & CMA (conservative investment) returns are not included since hard to replicate with ETFs
    
    the risk free rate is calculated using the US1M t-bill as a proxy (for USD-denominated assets)

    
    """

    def __init__(self, factor_source: str = "french", start_date = None, end_date = None, return_type:str = 'simple', frequency:str = 'daily', hac="auto"):

        #if start date is None, API will pull from the oldest date possible of all data sources
        #if end date is None, API will pull till the latest possible date of all data sources

        if factor_source not in {'french','etf'}:
            raise ValueError("input proper factor_source: 'french' (default) or 'etf'")

        if return_type not in {'simple','log'}:
            raise ValueError("input proper return_type: 'log' or 'simple'")

        if frequency not in {'daily','weekly','monthly','annually'}:
            raise ValueError("available input frequency: 'daily', 'weekly', 'monthly', 'annually'")

        freq_dict = {
            "daily": "d",
            "weekly": "w",
            "monthly": "m",
            "annually": "a"
        }

        
       

        #self.asset = asset.lower()
        self.return_type = return_type
        self.factor_source = factor_source

        self.freq = frequency
        self.tiingo_freq = frequency
        self.fred_freq = freq_dict[frequency]

        self.start = start_date
        self.end = end_date
        # hac : heteroscedasticity and autocorrelation robust (HAC) using n lags
        #
        # this is for the regression;
        #
        # users can override their hac lag number, but default will be automatic
        self.hac_lags = self._resolve_hac_lags(hac)

    def asset_list(self, *args:str):

        self.assets = []
        for ticker in args:
            if type(ticker) != str:
                raise ValueError("asset tickers can only be in string format!")
            self.assets.append(ticker.lower())

        return

    def regress(self):
        #the actual API the user uses
        
        # how it works:
        # step1: gathers the simple and log returns df for each asset in the self.assets list, then stores the pair 'ticker: df' in assets_df_dic; will drop the column of whichever type of return that is not needed.
        #
        # step2: gets the fama-french factor data df; each column calculated based on whether simple or log returns wanted
        #
        # step3: merges each asset return df in the assets_df_dic with the french factor data df after aligning the data points for daily/monthly/annually returns, then calculates the excess asset returns. then stores the asset: merged_df in merged_df_dic
        #
        # step4: for each merged_df in merged_df_dic, do the OLS regression of excess asset against the five factors returns with statsmodel with the set HAC maxlags and store the model results in results dic


        # this dic will store all relevant information and regression results
        grand_results = {}
        grand_results['basic_information'] = {
            'observations_frequency': self.freq,
            'return_convention': self.return_type,
            'factor_data_source': self.factor_source,
            'HAC_lags': self.hac_lags,
            'asset_list': self.assets.copy()
        }
        

        if self.factor_source == 'etf':
            excess_df = self._get_asset_excess_returns()

            market_fac, smb_fac, hml_fac, mtum_fac, rmw_fac = self._get_factor_returns()

            self._merge_factors()

        if self.factor_source == 'french':
            if self.freq not in {'daily','monthly','annually'}:
                raise ValueError("using french factor data is only available in input frequency: 'daily', OR 'monthly' OR 'annually'; otherwise use 'etf' as factor_source")

            if self.freq == 'daily':
                french_factor_df = self._load_french('daily').copy()
            elif self.freq == 'monthly':
                french_factor_df = self._load_french('monthly').copy()
            else:
                french_factor_df = self._load_french('annually').copy()

            grand_results['french_data'] = french_factor_df
            grand_results['assets_data'] = {}

            assets_df_dic = {}

            for ticker in self.assets:
                asset_returns_df = self._get_tiingo_df(ticker).copy()
                assets_df_dic[ticker] = asset_returns_df

                grand_results['assets_data'][ticker] = asset_returns_df
                

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
                    assets_df_dic[ticker] = df[['date', f'{ticker}_log_returns']].copy()


            else:
                for ticker, df in assets_df_dic.items():
                    assets_df_dic[ticker] = df[['date', f'{ticker}_simple_returns']].copy()


          

            grand_results['merged_data'] = {}

            if self.freq == 'daily':
                
                merged_df_dic = {}
                
                for ticker, df in assets_df_dic.items():
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
                        f'{ticker} data window': f'{asset_start} to {asset_end}',
                        f'french factor data window': f'{ff_start} to {ff_end}',
                        f'regression data window': f'{actual_start} to {actual_end}'
                    }

                
            elif self.freq == 'monthly':

                merged_df_dic = {}
                french_factor_df = french_factor_df.copy()
                french_factor_df["__period"] = (french_factor_df["date"].dt.to_period("M"))

                for ticker, df in assets_df_dic.items():
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
                          f'{actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}'
                    )

                    grand_results['merged_data'][ticker] = {
                        f'{ticker} data window': f'{asset_start} to {asset_end}',
                        f'french factor data window': f'{ff_start} to {ff_end}',
                        f'regression data window': f'{actual_start} to {actual_end}'
                    }
                
                
            elif self.freq == 'annually':

                
                merged_df_dic = {}
                french_factor_df = french_factor_df.copy()
                french_factor_df["__period"] = (french_factor_df["date"].dt.year)

                for ticker, df in assets_df_dic.items():
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
                        f'{ticker} data window': f'{asset_start} to {asset_end}',
                        f'french factor data window': f'{ff_start} to {ff_end}',
                        f'regression data window': f'{actual_start} to {actual_end}'
                    }



            #   creating a column for asset excess returns
            if self.return_type == "simple":

                for ticker, df in merged_df_dic.items():
                    df[f'{ticker}-RF'] = df[f'{ticker}_simple_returns'] - df['RF']

                    grand_results['merged_data'][ticker]['merged_data_dataframe'] = df.copy()
            else:
                for ticker, df in merged_df_dic.items():
                    df[f'{ticker}-RF'] = df[f'{ticker}_log_returns'] - df['RF']

                    grand_results['merged_data'][ticker]['merged_data_dataframe'] = df.copy()




            # OLS regression

            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
            results = {}

            grand_results['regression_results'] = {}

            for ticker, df in merged_df_dic.items():
                # Dependent variable: asset excess return
                y = df[f'{ticker}-RF']

                # Independent variables: Fama-French factors
                X = df[
                    ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
                ]

                # Add intercept (alpha)
                X = sm.add_constant(df[factor_cols])

                if self.hac_lags is None:
                    model = sm.OLS(y, X, missing="drop").fit()
                else:
                    model = sm.OLS(y, X, missing="drop").fit(
                        cov_type="HAC",
                        cov_kwds={"maxlags": self.hac_lags}
                    )

                results[ticker] = model

                grand_results['regression_results'][ticker] = {
                    "model": {
                        "name": "Fama-French 5 Factor",
                        "type": "OLS",
                        "dependent_variable": f"{ticker}-RF",
                        "factors": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
                    },
                    "sample": {
                        "frequency": f"{self.freq}",
                        "start_date": f"{actual_start}",
                        "end_date": f"{actual_end}",
                        "n_observations": int(model.nobs),
                    },
                    "specification": {
                        "dependent_variable": f"{ticker}-RF",
                        "independent_variables": ["Mkt-RF", "SMB", "HML", "RMW","CMA"],
                        "intercept": True,
                    },
                    "inference": {
                        "covariance_type": f'{"HAC" if self.hac_lags > 0 else 'Standard'}',
                        "hac_maxlags": f'{self.hac_lags}',
                        "hac_selection": "frequency_default"
                    },
                    "coefficients": {
                        "const (alpha)": model.params['const'].item(),
                        "Mkt-RF (market-beta)": model.params['Mkt-RF'].item(),
                        "SMB (size-exposure)": model.params["SMB"].item(),
                        "HML (value vs growth)": model.params["HML"].item(),
                        "RMW (profitability-exposure)": model.params["RMW"].item(),
                        "CMA (conservative vs aggressive investment)": model.params["CMA"].item(),
                    },

                    "standard_errors": {
                        "const": model.bse['const'].item(),
                        "Mkt-RF": model.bse['Mkt-RF'].item(),
                        "SMB": model.bse['SMB'].item(),
                        "HML": model.bse['HML'].item(),
                        "RMW": model.bse['RMW'].item(),
                        "CMA": model.bse['CMA'].item(),
                    },

                    "t_statistics": {
                        "const": model.tvalues['const'].item(),
                        "Mkt-RF": model.tvalues['Mkt-RF'].item(),
                        "SMB": model.tvalues['SMB'].item(),
                        "HML": model.tvalues['HML'].item(),
                        "RMW": model.tvalues['RMW'].item(),
                        "CMA": model.tvalues['CMA'].item(),
                    },

                    "p_values": {
                        "const": model.pvalues['const'].item(),
                        "Mkt-RF": model.pvalues['Mkt-RF'].item(),
                        "SMB": model.pvalues['SMB'].item(),
                        "HML": model.pvalues['HML'].item(),
                        "RMW": model.pvalues['RMW'].item(),
                        "CMA": model.pvalues['CMA'].item(),
                    },
                }

            # for model in results.values():
            #     print(model.summary())

            return grand_results

    def _get_tiingo_df(self, ticker):
        tiingo_api_key = os.getenv('TIINGO_API_KEY')
        tiingo_obj = TiingoApi(tiingo_api_key, self.tiingo_freq, True)
        df = tiingo_obj.get_data(ticker, self.start, self.end).copy()

        df[f'{ticker}_simple_returns'] = (df['adjClose'] / df['adjClose'].shift(1)) - 1
        df[f'{ticker}_log_returns'] = np.log(df['adjClose'] / df['adjClose'].shift(1))
        df = df[['date', f'{ticker}_log_returns', f'{ticker}_simple_returns']]

        df.dropna(ignore_index=True, inplace=True)

        return df

    def _get_rf_df(self):
        rf_ticker = "dgs3mo"
        fred_api_key = os.getenv('FRED_API_KEY')
        fred_obj = FredApi(fred_api_key, self.fred_freq)
        rf_df = fred_obj.get_data(rf_ticker,self.start, self.end)

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

        return rf_df
        
    def _get_asset_excess_returns(self):
        
        # tiingo_api_key = os.getenv('TIINGO_API_KEY')
        # tiingo_obj = TiingoApi(tiingo_api_key, self.tiingo_freq, True)
        # asset_df = tiingo_obj.get_data(self.asset, self.start, self.end)

        # rf_ticker = "dgs3mo"
        # fred_api_key = os.getenv('FRED_API_KEY')
        # fred_obj = FredApi(fred_api_key, self.fred_freq)
        # rf_df = fred_obj.get_data(rf_ticker,self.start, self.end)
        

        
        # asset_df[f'{self.asset}_simple_returns'] = (asset_df['adjClose'] / asset_df['adjClose'].shift(1)) - 1
        # asset_df[f'{self.asset}_log_returns'] = np.log(asset_df['adjClose'] / asset_df['adjClose'].shift(1))
        # asset_df = asset_df[['date', f'{self.asset}_log_returns', f'{self.asset}_simple_returns']]

        asset_df = self._get_tiingo_df(self.asset)
        rf_df = self._get_rf_df()
        
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

        # rf_df["days"] = rf_df["date"].diff().dt.days

        # rf_df["rf_simple_returns"] = (
        #     rf_df["value"].shift(1) / 100
        #     * rf_df["days"]
        #     / 365
        # )
        # rf_df["rf_log_returns"] = np.log1p(rf_df["rf_simple_returns"])


        
        # #rf_return_df = rf_return_df.dropna().reset_index(drop=True)

        excess_df = asset_df.merge(
            rf_df[["date", "rf_log_returns", "rf_simple_returns"]],
            on="date",
            how="inner"
        )

        excess_df[f"{self.asset}_excess_simple_return"] = (
            excess_df[f'{self.asset}_simple_returns'] - excess_df["rf_simple_returns"]
        )

        excess_df[f"{self.asset}_excess_log_return"] = (
            excess_df[f'{self.asset}_log_returns'] - excess_df["rf_log_returns"]
        )

        excess_df = excess_df[['date',f'{self.asset}_excess_simple_return',f'{self.asset}_excess_log_return']]

        excess_df.dropna().reset_index(drop=True)

        return excess_df

    def _merge(df1, df2):
        df = df1.merge(
            df2,
            on="date",
            how="inner"
        )

        return df

    def _get_factor_returns(self):

        #market excess
        market_df = self._get_tiingo_df("spy")
        rf_df = self._get_rf_df()

        market_excess = market_df.merge(
            rf_df[["date", "rf_log_returns", "rf_simple_returns"]],
            on="date",
            how="inner"
        )
        
        market_excess[f"spy_excess_simple_return"] = (
            market_excess['spy_simple_returns'] - market_excess["rf_simple_returns"]
        )
        
        market_excess[f"spy_excess_log_return"] = (
            market_excess[f'spy_log_returns'] - market_excess["rf_log_returns"]
        )
        
        market_excess = market_excess[['date',f'spy_excess_simple_return',f'spy_excess_log_return']]
        
        market_excess.dropna().reset_index(drop=True)
        

        #smb
        small_df = self._get_tiingo_df("iwm")
        large_df = self._get_tiingo_df("spy")

        smb = self._merge(small_df,large_df)
        smb['smb_simple_return'] = (
            smb['iwm_simple_returns'] - smb['spy_simple_returns']
        )

        smb['smb_log_return'] = (
                    smb['iwm_log_returns'] - smb['spy_log_returns']
                )
        smb = smb[['date','smb_simple_return','smb_log_return']]
        smb.dropna().reset_index(drop=True)

        #hml
        value_df = self._get_tiingo_df("iwd")
        growth_df = self._get_tiingo_df("iwf")

        hml = self._merge(value_df,growth_df)

        hml['hml_simple_return'] = (
            hml['iwd_simple_returns'] - hml['iwf_simple_returns']
        )
        hml['hml_log_return'] = (
            hml['iwd_log_returns'] - hml['iwf_log_returns']
        )
        hml = hml[['date','hml_simple_return','hml_log_return']]
        hml.dropna().reset_index(drop=True)


        #momentum
        mtum_df = self._get_tiingo_df("mtum")
        market_df = self._get_tiingo_df("spy")

        mtum = self._merge(mtum_df, market_df)
        
        mtum['mtum_simple_return'] = (
            mtum['mtum_simple_returns'] - mtum['spy_simple_returns']
        )
        mtum['mtum_log_return'] = (
            mtum['mtum_log_returns'] - mtum['spy_log_returns']
        )
        mtum = mtum[['date','mtum_simple_return','mtum_log_return']]
        mtum.dropna().reset_index(drop=True)

        #RMW
        robust_df = self._get_tiingo_df("qual")
        market_df = self._get_tiingo_df("spy")
        
        rmw = self._merge(robust_df, market_df)
        rmw['rmw_simple_return'] = (
            rmw['qual_simple_returns'] - rmw['spy_simple_returns']
        )
        rmw['rmw_log_return'] = (
            rmw['qual_log_returns'] - rmw['spy_log_returns']
        )
        rmw = rmw[['date','mtum_simple_return','mtum_log_return']]
        rmw.dropna().reset_index(drop=True)

        return market_excess, smb, hml, mtum, rmw
        



        

        
        rf_ticker = "dgs3mo"
        fred_api_key = os.getenv('FRED_API_KEY')
        fred_obj = FredApi(fred_api_key, self.fred_freq)
        rf_df = fred_obj.get_data(rf_ticker,self.start, self.end)
        

        market_df[f'{self.asset}_simple_returns'] = (market_df['adjClose'] / asset_df['adjClose'].shift(1)) - 1
        asset_df[f'{self.asset}_log_returns'] = np.log(asset_df['adjClose'] / asset_df['adjClose'].shift(1))
        asset_df = asset_df[['date', f'{self.asset}_log_returns', f'{self.asset}_simple_returns']]
        
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


        
        #rf_return_df = rf_return_df.dropna().reset_index(drop=True)

        excess_df = asset_df.merge(
            rf_df[["date", "rf_log_returns", "rf_simple_returns"]],
            on="date",
            how="inner"
        )

        excess_df[f"{self.asset}_excess_simple_return"] = (
            excess_df[f'{self.asset}_simple_returns'] - excess_df["rf_simple_returns"]
        )

        excess_df[f"{self.asset}_excess_log_return"] = (
            excess_df[f'{self.asset}_log_returns'] - excess_df["rf_log_returns"]
        )

        excess_df = excess_df[['date',f'{self.asset}_excess_simple_return',f'{self.asset}_excess_log_return']]

        excess_df.dropna().reset_index(drop=True)

        return excess_df
        
    def _merge_factors(self, asset, market, smb, hml, mtum, rmw):
        pass

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


if __name__ == "__main__":
    fac = EquityFactorsRegression(factor_source='french', frequency='daily')
    asset_list = ['msft','nvda']
    fac.asset_list(*asset_list)
    data = fac.regress()

    for asset in asset_list:
        pprint(data['regression_results'][asset], sort_dicts=False, width=1, indent=4)
