import numpy as np
import pandas as pd
import statsmodels.api as sm
import regression_beta.returns as returns

class OLSRegression():

    def __init__(self, asset1: pd.DataFrame, asset2: pd.DataFrame, return_type: str ="log"):
        col = f"{return_type}-returns"

        #merge the two df by timeframes with how=inner so that the returns series will start on the latest timestamps which both assets have
        merged = pd.merge(
            asset1,
            asset2,
            on="date",
            how="inner",
            suffixes=("_1", "_2")
            )

        merged = merged.dropna(
            subset=[f"{col}_1", f"{col}_2"]
            ).reset_index(drop=True)
        
        #hiding the time part of the pd datetime object
        merged = merged.sort_values('date').reset_index(drop=True)
        merged.style.format({
            'date': '%Y-%m-%d'
            })


        self.merged_return_series = merged.copy()

        
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

        merged = asset1[["date", col]].copy()
        merged = merged.rename(columns = {col :'y'})
 
        assets_names = []
 
        for asset_name, asset_df in assets.items():
            if col not in asset_df.columns:
                raise ValueError(f"factor '{asset_name}' is missing column '{col}' — did you pass returns, not prices?")
            df = asset_df[["date", col]].copy()
            df = df.rename(columns={col: asset_name})
            merged = pd.merge(merged, df, on="date", how="inner")
            assets_names.append(asset_name)
        #inner-merging sequentially keeps only timestamps common to the dependent asset and every factor
 
        if merged.empty:
            raise ValueError("no overlapping timestamps across dependent asset and all factors")
 
        self.assets_names = assets_names
        self.y = merged["y"]
        self.x = merged[assets_names]

        #hiding the time part of the pd datetime object
        merged = merged.sort_values('date').reset_index(drop=True)
        merged.style.format({
            'date': '%Y-%m-%d'
        })

        
        self.merged_return_series = merged.copy()
 
    def ols(self):
        x = sm.add_constant(self.x, has_constant="add")
        model = sm.OLS(self.y, x).fit()
        return model
    

if __name__ == "__main__":
    pass