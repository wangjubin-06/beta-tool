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
from regression_beta.returns import simple_returns, log_returns
import requests
from io import BytesIO, StringIO
import zipfile


class EquityFactorsRegression():
    """
    
    this class produces an object that shows the factor betas of a particular asset
    
    the betas comes from regressing the asset's equity premium (asset return - risk free rate) against market (rm - rf) returns, SMB (small-caps) returns, HML (value) returns, MOM (momentum) returns, RMW (quality)
    
    
    the returns for each factor are proxied using ETFs, instead of actual Fama-French factor returns for simplicity. a new version with the actual returns may come.

    RMW (quality) returns & CMA (conservative investment) returns are not included since hard to replicate with ETFs
    
    the risk free rate is calculated using the US1M t-bill as a proxy (for USD-denominated assets)

    
    """

    def __init__(self, factor_source: str = "french", start_date = None, end_date = None, return_type:str = 'simple', frequency:str = 'daily'):

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


        if self.factor_source == 'etf':
            excess_df = self._get_asset_excess_returns()

            market_fac, smb_fac, hml_fac, mtum_fac, rmw_fac = self._get_factor_returns()

            self._merge_factors()

        elif self.factor_source == 'french':
            if self.freq not in {'daily','monthly','annually'}:
                raise ValueError("using french factor data is only available in input frequency: 'daily', OR 'monthly' OR 'annually'; otherwise use 'etf' as factor_source")


            #ticker = self.ticker

            if self.freq == 'daily':
                french_factor_df = self._load_french('daily')
            elif self.freq == 'monthly':
                french_factor_df = self._load_french('monthly')
            else:
                french_factor_df = self._load_french('annually')

            assets_df_dic = {}

            for ticker in self.assets:
                asset_returns_df = self._get_tiingo_df(ticker)
                assets_df_dic[ticker] = asset_returns_df



                #######

                print(ticker, "\n", asset_returns_df.tail())
                
                #######
                


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
                    assets_df_dic[ticker] = df[['date', f'{ticker}_log_returns']]


            else:
                for ticker, df in assets_df_dic.items():
                    assets_df_dic[ticker] = df[['date', f'{ticker}_simple_returns']]

                #asset_returns_df = asset_returns_df[['date', f'{ticker}_simple_returns']]

          

            ######
            print(french_factor_df.tail())
            ######



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
                    print(
                        f"{ticker}: asset data {asset_start:%Y-%m-%d} → "
                        f"{asset_end:%Y-%m-%d};\n"
                        f"FF data {ff_start:%Y-%m-%d} → {ff_end:%Y-%m-%d};\n"
                        f"regression observations: {actual_start:%Y-%m-%d} → {actual_end:%Y-%m-%d};\n"
                    )

                
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


          


            #   creating a column for asset excess returns
            if self.return_type == "simple":

                for ticker, df in merged_df_dic.items():
                    df[f'{ticker}-RF'] = df[f'{ticker}_simple_returns'] - df['RF']
            else:
                for ticker, df in merged_df_dic.items():
                    df[f'{ticker}-RF'] = df[f'{ticker}_log_returns'] - df['RF']



            ### TESTING
            for ticker, df in merged_df_dic.items():
                print(ticker,'\n\n', df.head())
            ###


            # OLS regression

            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
            results = {}

            for ticker, df in merged_df_dic.items():
                # Dependent variable: asset excess return
                y = df[f'{ticker}-RF']

                # Independent variables: Fama-French factors
                X = df[
                    ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
                ]

                # Add intercept (alpha)
                X = sm.add_constant(df[factor_cols])

                model = sm.OLS(y, X, missing="drop").fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": 3}
                )

                results[ticker] = model

            for model in results.values():
                print(model.summary())


    def _get_tiingo_df(self, ticker):
        tiingo_api_key = os.getenv('TIINGO_API_KEY')
        tiingo_obj = TiingoApi(tiingo_api_key, self.tiingo_freq, True)
        df = tiingo_obj.get_data(ticker, self.start, self.end)

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

        # WARNING: THE default dataset here is US factor data
        #
        # only use US factor data for regressing US equities
        #
        # for OTHER regions, get factor data for other regions (important!) (coming soon!)
        #
        #

        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip" if frequency == 'daily' else "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

        if frequency == 'daily':
            response = requests.get(url)
            response.raise_for_status()

            with zipfile.ZipFile(BytesIO(response.content)) as z:
                filename = "F-F_Research_Data_5_Factors_2x3_daily.csv"

                with z.open(filename) as f:
                    df = pd.read_csv(
                        f,
                        skiprows=3,
                        skipfooter=1,
                        engine="python"
                    )

            df = df.rename(columns={df.columns[0]: "date"})

            # Convert Date to datetime
            df["date"] = pd.to_datetime(
                df["date"].astype(str),
                format="%Y%m%d"
            )

            # Fama-French returns are in percent; convert to decimals
            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
            df[factor_cols] /= 100

            return df
        
        elif frequency == 'monthly':
            response = requests.get(url)
            response.raise_for_status()

            with zipfile.ZipFile(BytesIO(response.content)) as z:
                filename = "F-F_Research_Data_5_Factors_2x3.csv"

                with z.open(filename) as f:
                    text = f.read().decode("utf-8")

            # Keep only the monthly section
            monthly_text = text.split("Annual Factors:", 1)[0]

            df = pd.read_csv(
                StringIO(monthly_text),
                skiprows=3
            )

            df = df.rename(columns={df.columns[0]: "date"})

            # Convert Date to datetime
            df["date"] = (
                pd.to_datetime(df["date"].astype(str), format="%Y%m")
                + pd.offsets.BMonthEnd(0)
            )

            # Fama-French returns are in percent; convert to decimals
            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
            df[factor_cols] /= 100

            return df
        else:
            response = requests.get(url)
            response.raise_for_status()
        
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                filename = "F-F_Research_Data_5_Factors_2x3.csv"
        
                with z.open(filename) as f:
                    text = f.read().decode("utf-8")
        
            # Keep only the annual section
            annual_text = text.split("Annual Factors:", 1)[1]
        
            df = pd.read_csv(
                StringIO(annual_text),
                skiprows=1,
                skipfooter=2
            )
        
            df = df.rename(columns={df.columns[0]: "date"})
        
            # Convert Date to datetime
            df["date"] = (
                pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y")
                + pd.offsets.BYearEnd(0)
            )
        
            # Fama-French returns are in percent; convert to decimals
            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
            df[factor_cols] /= 100
        
            return df



    # def _excess_asset_returns_df(self):
    #     asset_returns = self._get_returns_df(
    #         self.asset,
    #         self._get_prices_df(self.asset, self.start_date, self.interval),
    #         self.return_type
    #     )

    #     asset_excess = asset_returns.merge(
    #         self.risk_free_df,
    #         on="timestamp",
    #         how="inner",
    #         suffixes=("_asset", "_market")
    #         )
        
    #     asset_excess["asset-excess-returns"] = asset_excess[f"{self.asset} adjusted_close_asset"] - asset_excess[f'daily-rf-{self.return_type}-returns_rf']
    #     asset_excess = asset_excess[["timestamp", "asset-excess-returns"]]

        
    #     self.asset_excess_returns_df = asset_excess.copy()


        

    # @staticmethod
    # def start_date_from_today_lookback(lookback:str):
    #     lookback_dict ={
    #         '1y': 1,
    #         '2y': 2,
    #         '3y': 3,
    #         '5y': 5,
    #         '10y': 10,
    #         '20y': 20,
    #         '30y': 30
    #     }
    #     # Get today's date
    #     today = date.today()

    #     try:
    #         past_date = today.replace(year=today.year - lookback_dict[lookback])
    #     except ValueError:
    #         # Handle the edge case if today is Feb 29 and 5 years ago wasn't a leap year
    #         past_date = today.replace(year=today.year - lookback_dict[lookback], day=28)

    #     # Convert to 'yyyy-mm-dd' string format
    #     result_string = past_date.isoformat()
    #     return result_string




    # @staticmethod
    # def _get_returns_df(ticker, price_df, return_type:str = "log"):
    #     """
    #     this method returns a cleaned-up df of asset returns; used for asset and factor etfs returns

    #     """
    #     if return_type == "log":
    #         return_df = log_returns(price_df)
    #     else:
    #         return_df = simple_returns(price_df)

    #     return_df = return_df.loc[:, ['timestamp', 'adjusted_close']].copy()
    #     return_df.rename(columns={'adjusted_close': f"{ticker} adjusted_close"}, inplace=True)

    #     return return_df

    # @staticmethod
    # def _get_prices_df(ticker:str, start_date, interval):
    #     """
    #     this method returns df of asset prices; used for asset and factor etfs
        
    #     """
    #     df = AssetData(ticker=ticker, start_date=start_date, interval=interval).get_prices()
    #     return df


    # def _get_factor_etf_df(self):
    #     self.market_return_df = self._get_returns_df(
    #         "spy",
    #         self._get_prices_df("spy", self.start_date, self.interval),
    #         self.return_type
    #     )

    #     self.smallcap_return_df = self._get_returns_df(
    #         "iwm",
    #         self._get_prices_df("iwm", self.start_date, self.interval),
    #         self.return_type
    #     )

    #     self.value_return_df = self._get_returns_df(
    #         "iwd",
    #         self._get_prices_df("iwd", self.start_date, self.interval),
    #         self.return_type
    #     )

    #     self.growth_return_df = self._get_returns_df(
    #         "iwf",
    #         self._get_prices_df("iwf", self.start_date, self.interval),
    #         self.return_type
    #     )

    #     self.quality_return_df = self._get_returns_df(
    #         "qual",
    #         self._get_prices_df("qual", self.start_date, self.interval),
    #         self.return_type
    #     )

    #     self.momentum_return_df = self._get_returns_df(
    #         "mtum",
    #         self._get_prices_df("mtum", self.start_date, self.interval),
    #         self.return_type
    #     )



    # def _excess_factor_returns_df(self):

    #     market_excess = self.market_return_df.merge(
    #         self.risk_free_df,
    #         on="timestamp",
    #         how="inner",
    #         suffixes=("_market", "_rf")
    #         )

    #     market_excess["market-excess-returns"] = market_excess["spy adjusted_close_market"] - market_excess[f'daily-rf-{self.return_type}-returns_rf']
    #     market_excess = market_excess[["timestamp", "market-excess-returns"]]

    #     self.market_excess_returns_df = market_excess.copy()

    #     smb = self.smallcap_return_df.merge(
    #         self.market_return_df,
    #         on="timestamp",
    #         how="inner",
    #         suffixes=("_small", "_market")
    #         )
        
    #     smb["SMB"] = smb["iwm adjusted_close_small"] - smb["spy adjusted_close_market"]
    #     smb = smb[["timestamp", "SMB"]]
        
    #     self.smb_returns_df = smb.copy()
        

    #     hml = self.value_return_df.merge(
    #         self.growth_return_df,
    #         on="timestamp",
    #         how="inner",
    #         suffixes=("_value", "_growth")
    #         )

    #     hml["HML-returns"] = hml["iwd adjusted_close_value"] - hml["iwf adjusted_close_growth"]
    #     hml = hml[["timestamp", "HML"]]

    #     self.hml_returns_df = hml.copy()


    #     qual = self.quality_return_df.merge(
    #         self.market_return_df,
    #         on="timestamp",
    #         how="inner",
    #         suffixes=("_qual", "_market")
    #         )
        
    #     qual["RMW"] = qual["qual adjusted_close_qual"] - qual["spy adjusted_close_market"]
    #     qual = qual[["timestamp", "RMW"]]
        
    #     self.rmw_returns_df = qual.copy()

    #     mtum = self.momentum_return_df.merge(
    #         self.market_return_df,
    #         on="timestamp",
    #         how="inner",
    #         suffixes=("_mtum", "_market")
    #         )
        
    #     mtum["momentum"] = mtum["mtum adjusted_close_mtum"] - mtum["spy adjusted_close_market"]
    #     mtum = mtum[["timestamp", "momentum"]]
        
    #     self.mtum_returns_df = mtum.copy()



    # def _merge_df(self):
    #     dfs = [
    #         self.asset_excess_returns_df,
    #         self.market_excess_returns_df,
    #         self.smb_returns_df,
    #         self.hml_returns_df,
    #         self.rmw_returns_df,
    #         self.mtum_returns_df
    #     ]

    #     merged = dfs[0].copy()
    #     for df in dfs[1:]:
    #         merged = merged.merge(df, on="timestamp", how="inner")

    #     merged = merged.sort_values("timestamp").reset_index(drop=True)

    #     self.summary = merged
        


if __name__ == "__main__":
    # asset_prices = AssetData(ticker="spy",start_date="1999-01-01",interval="daily").get_prices()
    # return_fn = log_returns
    # asset_returns = return_fn(asset_prices)
    # asset_returns = asset_returns.loc[:, ['timestamp', 'adjusted_close']].copy()
    # asset_returns.rename(columns={'adjusted_close': "SPY adjusted_close"}, inplace=True)
    # print(asset_returns.head())

    # my_fac_obj = EquityFactorsRegression("aapl")
    # print(my_fac_obj.summary.head())

    # url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    # response = requests.get(url)
    # response.raise_for_status()

    # with zipfile.ZipFile(BytesIO(response.content)) as z:
    #     filename = "F-F_Research_Data_5_Factors_2x3.csv"

    #     with z.open(filename) as f:
    #         text = f.read().decode("utf-8")

    # # Keep only the monthly section
    # monthly_text = text.split("Annual Factors:", 1)[1]

    # df = pd.read_csv(
    #     StringIO(monthly_text),
    #     skiprows=1,
    #     skipfooter=2
    # )

    # df = df.rename(columns={df.columns[0]: "date"})

    # # Convert Date to datetime
    # df["date"] = (
    #     pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y")
    #     + pd.offsets.BYearEnd(0)
    # )

    # # Fama-French returns are in percent; convert to decimals
    # factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    # df[factor_cols] /= 100
    #print(df.head())

    fac = EquityFactorsRegression('french','2025-01-01')
    fac.asset_list('aapl','goog')
    fac.regress()
