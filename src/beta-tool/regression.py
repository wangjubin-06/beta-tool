import numpy as np
import pandas as pd
import statsmodels.api as sm
import data, returns

class OLSRegression():

    def __init__(self, asset1: pd.DataFrame, asset2: pd.DataFrame, return_type: str ="log"):
        col = f"{return_type}-returns"

        #merge the two df by timeframes with how=inner so that the returns series will start on the latest timestamps which both assets have
        merged = pd.merge(
            asset1,
            asset2,
            on="timestamp",
            how="inner",
            suffixes=("_1", "_2")
            )
        
        #hiding the time part of the pd datetime object
        merged['timestamp'] = merged['timestamp'].dt.strftime('%Y-%m-%d')


        self.merged_return_series = merged

        
        self.x = merged[f"{col}_2"]
        self.y = merged[f"{col}_1"]

    def ols(self):
        x = sm.add_constant(self.x, has_constant="add")
        model = sm.OLS(self.y,x).fit()
        return model

class MultiFactorRegression():
    """
    Generic OLS regression of one asset's returns against N other assets' returns
    at once. Not tied to any fixed factor set (e.g. Fama-French) — the
    "factors" here are just any other assets' return series the user supplies.
    """
    def __init__(self, asset1: pd.DataFrame, assets: dict[str, pd.DataFrame], return_type: str = "log"):

        col = f"{return_type}-returns"
 
        if len(assets) < 1:
            raise ValueError("at least one factor asset is required")
 
        merged = asset1[["timestamp", col]].rename(columns={col: "y"})
        assets_names = []
 
        for asset_name, asset_df in assets.items():
            if col not in asset_df.columns:
                raise ValueError(f"factor '{asset_name}' is missing column '{col}' — did you pass returns, not prices?")
            renamed = asset_df[["timestamp", col]].rename(columns={col: asset_name})
            merged = pd.merge(merged, renamed, on="timestamp", how="inner")
            assets_names.append(asset_name)
        #inner-merging sequentially keeps only timestamps common to the dependent asset and every factor
 
        if merged.empty:
            raise ValueError("no overlapping timestamps across dependent asset and all factors")
 
        self.assets_names = assets_names
        self.y = merged["y"]
        self.x = merged[assets_names]
 
    def ols(self):
        x = sm.add_constant(self.x, has_constant="add")
        model = sm.OLS(self.y, x).fit()
        return model
    

if __name__ == "__main__":
    apple = data.AssetData("aapl","2y","daily")
    appledata = apple.get_prices()

    msft = data.AssetData("msft","2y","daily")
    msftdata = msft.get_prices()

    apple_returns = returns.log_returns(appledata)
    msft_returns = returns.log_returns(msftdata)

    regressobj = OLSRegression(apple_returns,msft_returns)
    results = regressobj.ols()

    print(results.params)
    print(results.params.index)

