import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import AssetData
from returns import log_returns, simple_returns
from riskfreerate import generate_risk_free_rate_df
from datetime import date


class EquityFactorsRegression():
    """
    
    this class produces an object that shows the factor betas of a particular asset
    
    the betas comes from regressing the asset's equity premium (asset return - risk free rate) against market (rm - rf) returns, SMB (small-caps) returns, HML (value) returns, MOM (momentum) returns, RMW (quality)
    
    
    the returns for each factor are proxied using ETFs, instead of actual Fama-French factor returns for simplicity. a new version with the actual returns may come.

    RMW (quality) returns & CMA (conservative investment) returns are not included since hard to replicate with ETFs
    
    the risk free rate is calculated using the US1M t-bill as a proxy (for USD-denominated assets)

    
    """

    def __init__(
            self,
            asset:str,
            return_type:str = 'log',
            lookback_period:str = 'max',
            interval:str = 'daily'
        ):

        if return_type not in {'simple','log'}:
            raise ValueError("select proper return_type: log or simple")

        interval = interval.lower()
        interval_dict = {
            "daily": "1d",
            "weekly": "1w",
            "monthly": "1mo",
            }
        if interval in interval_dict:
            self.interval = interval_dict[interval]
        elif interval in interval_dict.values():
            self.interval = interval
        else:
            raise ValueError("interval is not of the correct format. options available: daily, weekly, monthly")

        self.asset = asset
        self.return_type = return_type
        self.lookback_period = lookback_period

        if lookback_period == "max":
            self.start_date = "1800-01-01"
        else:
            self.start_date = self.start_date_from_today_lookback(lookback_period)

        self.risk_free_df = generate_risk_free_rate_df(start=self.start_date,return_type=return_type)

        self._get_factor_etf_df()
        self._excess_factor_returns_df()

        self._excess_asset_returns_df()

        self._merge_df()




    def _excess_asset_returns_df(self):
        asset_returns = self._get_returns_df(
            self.asset,
            self._get_prices_df(self.asset, self.start_date, self.interval),
            self.return_type
        )

        asset_excess = asset_returns.merge(
            self.risk_free_df,
            on="timestamp",
            how="inner",
            suffixes=("_asset", "_market")
            )
        
        asset_excess["asset-excess-returns"] = asset_excess[f"{self.asset} adjusted_close_asset"] - asset_excess[f'daily-rf-{self.return_type}-returns_rf']
        asset_excess = asset_excess[["timestamp", "asset-excess-returns"]]
        
        self.asset_excess_returns_df = asset_excess.copy()


        

    @staticmethod
    def start_date_from_today_lookback(lookback:str):
        lookback_dict ={
            '1y': 1,
            '2y': 2,
            '3y': 3,
            '5y': 5,
            '10y': 10,
            '20y': 20,
            '30y': 30
        }
        # Get today's date
        today = date.today()

        try:
            past_date = today.replace(year=today.year - lookback_dict[lookback])
        except ValueError:
            # Handle the edge case if today is Feb 29 and 5 years ago wasn't a leap year
            past_date = today.replace(year=today.year - lookback_dict[lookback], day=28)

        # Convert to 'yyyy-mm-dd' string format
        result_string = past_date.isoformat()
        return result_string




    @staticmethod
    def _get_returns_df(ticker, price_df, return_type:str = "log"):
        """
        this method returns a cleaned-up df of asset returns; used for asset and factor etfs returns

        """
        if return_type == "log":
            return_df = log_returns(price_df)
        else:
            return_df = simple_returns(price_df)

        return_df = return_df.loc[:, ['timestamp', 'adjusted_close']].copy()
        return_df.rename(columns={'adjusted_close': f"{ticker} adjusted_close"}, inplace=True)

        return return_df

    @staticmethod
    def _get_prices_df(ticker:str, start_date, interval):
        """
        this method returns df of asset prices; used for asset and factor etfs
        
        """
        df = AssetData(ticker=ticker, start_date=start_date, interval=interval).get_prices()
        return df


    def _get_factor_etf_df(self):
        self.market_return_df = self._get_returns_df(
            "spy",
            self._get_prices_df("spy", self.start_date, self.interval),
            self.return_type
        )

        self.smallcap_return_df = self._get_returns_df(
            "iwm",
            self._get_prices_df("iwm", self.start_date, self.interval),
            self.return_type
        )

        self.value_return_df = self._get_returns_df(
            "iwd",
            self._get_prices_df("iwd", self.start_date, self.interval),
            self.return_type
        )

        self.growth_return_df = self._get_returns_df(
            "iwf",
            self._get_prices_df("iwf", self.start_date, self.interval),
            self.return_type
        )

        self.quality_return_df = self._get_returns_df(
            "qual",
            self._get_prices_df("qual", self.start_date, self.interval),
            self.return_type
        )

        self.momentum_return_df = self._get_returns_df(
            "mtum",
            self._get_prices_df("mtum", self.start_date, self.interval),
            self.return_type
        )



    def _excess_factor_returns_df(self):

        market_excess = self.market_return_df.merge(
            self.risk_free_df,
            on="timestamp",
            how="inner",
            suffixes=("_market", "_rf")
            )

        market_excess["market-excess-returns"] = market_excess["spy adjusted_close_market"] - market_excess[f'daily-rf-{self.return_type}-returns_rf']
        market_excess = market_excess[["timestamp", "market-excess-returns"]]

        self.market_excess_returns_df = market_excess.copy()

        smb = self.smallcap_return_df.merge(
            self.market_return_df,
            on="timestamp",
            how="inner",
            suffixes=("_small", "_market")
            )
        
        smb["SMB"] = smb["iwm adjusted_close_small"] - smb["spy adjusted_close_market"]
        smb = smb[["timestamp", "SMB"]]
        
        self.smb_returns_df = smb.copy()
        

        hml = self.value_return_df.merge(
            self.growth_return_df,
            on="timestamp",
            how="inner",
            suffixes=("_value", "_growth")
            )

        hml["HML-returns"] = hml["iwd adjusted_close_value"] - hml["iwf adjusted_close_growth"]
        hml = hml[["timestamp", "HML"]]

        self.hml_returns_df = hml.copy()


        qual = self.quality_return_df.merge(
            self.market_return_df,
            on="timestamp",
            how="inner",
            suffixes=("_qual", "_market")
            )
        
        qual["RMW"] = qual["qual adjusted_close_qual"] - qual["spy adjusted_close_market"]
        qual = qual[["timestamp", "RMW"]]
        
        self.rmw_returns_df = qual.copy()

        mtum = self.momentum_return_df.merge(
            self.market_return_df,
            on="timestamp",
            how="inner",
            suffixes=("_mtum", "_market")
            )
        
        mtum["momentum"] = mtum["mtum adjusted_close_mtum"] - mtum["spy adjusted_close_market"]
        mtum = mtum[["timestamp", "momentum"]]
        
        self.mtum_returns_df = mtum.copy()



    def _merge_df(self):
        dfs = [
            self.asset_excess_returns_df,
            self.market_excess_returns_df,
            self.smb_returns_df,
            self.hml_returns_df,
            self.rmw_returns_df,
            self.mtum_returns_df
        ]

        merged = dfs[0].copy()
        for df in dfs[1:]:
            merged = merged.merge(df, on="timestamp", how="inner")

        merged = merged.sort_values("timestamp").reset_index(drop=True)

        self.summary = merged
        


if __name__ == "__main__":
    # asset_prices = AssetData(ticker="spy",start_date="1999-01-01",interval="daily").get_prices()
    # return_fn = log_returns
    # asset_returns = return_fn(asset_prices)
    # asset_returns = asset_returns.loc[:, ['timestamp', 'adjusted_close']].copy()
    # asset_returns.rename(columns={'adjusted_close': "SPY adjusted_close"}, inplace=True)
    # print(asset_returns.head())

    my_fac_obj = EquityFactorsRegression("aapl")
    print(my_fac_obj.summary.head())
