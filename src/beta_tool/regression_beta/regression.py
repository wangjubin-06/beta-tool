import numpy as np
import pandas as pd
import statsmodels.api as sm
import regression_beta.returns as returns

class OLSRegression:

    def __init__(self, asset1: pd.DataFrame, asset2: pd.DataFrame, asset_1_col, asset_2_col, frequency, return_type: str ="log"):
        
        if return_type not in {'log','simple'}:
            raise ValueError("return_type has to be either 'log' or 'simple'")


        self.freq = frequency

        y_col = asset_1_col
        x_col = asset_2_col

        self.y_col = y_col
        self.x_col = x_col

        self.return_type = return_type

        
        
        if not any(f"{y_col}" in col for col in asset1.columns):
            raise KeyError(f'No column named {y_col} in asset1 dataframe provided')
        if not any(f"{x_col}" in col for col in asset2.columns):
            raise KeyError(f'No column named {x_col} in asset2 dataframe provided')
        if not any("date" in col for col in asset1.columns):
            raise KeyError(f'No date column in asset1 dataframe provided')
        if not any("date" in col for col in asset2.columns):
            raise KeyError(f'No date column in asset2 dataframe provided')
        

        #merge the two df by timeframes with how=inner so that the returns series will start on the latest timestamps which both assets have
        merged = pd.merge(
            asset1,
            asset2,
            on="date",
            how="inner",
            )

        merged = merged.dropna()

        
        #hiding the time part of the pd datetime object
        merged = merged.sort_values('date').reset_index(drop=True)

        merged.style.format({
            'date': '%Y-%m-%d'
            })

        self.start_date = merged['date'].min()
        self.end_date = merged['date'].max()


        self.merged_return_series = merged.copy()

        
        self.x = merged[x_col]
        self.y = merged[y_col]


    def ols(self):
        x = sm.add_constant(self.x, has_constant="add")

        model = sm.OLS(self.y,x).fit()

        self.results = model

        return model


    def results_df(self):

        # Summarise results in a dataframe

        ols_df = pd.DataFrame(
            {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'frequency': self.freq,
                'return_type': self.return_type,
                "n_obs": self.results.nobs,
                'beta': self.results.params[self.x_col],
                'alpha': self.results.params['const'],
                "alpha_p_value": self.results.pvalues['const'],
                'beta_std_error': self.results.bse[self.x_col],
                'beta_tstat': self.results.tvalues[self.x_col],
                'beta_pvalue': self.results.pvalues[self.x_col],
                "beta_ci_low": self.results.conf_int().loc[self.x_col, 0],
                "beta_ci_high": self.results.conf_int().loc[self.x_col, 1],
                'r_squared': self.results.rsquared,
                'residual_volatility': np.sqrt(self.results.mse_resid),
            },
            index = [0]
        )

        # Annualized alpha
        ols_df['annualized_alpha'] = ((1 + ols_df['alpha']) ** 252 - 1) if self.return_type == 'simple' else (ols_df['alpha'] * 252)

        self.ols_df = ols_df

        return ols_df

    def summary(self, asset_1_name, asset_2_name):
        """Return a formatted summary of the regression results."""

        print("=" * 60)
        print(f" OLS Regression Summary")
        print(f"{asset_1_name} against {asset_2_name}")
        print("=" * 60)

        print(f"\nObservation period")
        print(f"  Start:              {self.ols_df['start_date'].item()}")
        print(f"  End:                {self.ols_df['end_date'].item()}")
        print(f"  Observations:       {int(self.ols_df['n_obs'].item())}")
        print(f"  Frequency:          {self.freq}")
        print(f"  Return type:        {self.return_type}")

        print(f"  Beta:               {float(self.ols_df['beta'].item()):.5f}")
        print(
            f"  95% CI:             "
            f"[{float(self.ols_df['beta_ci_low'].item()):.5f}, "
            f"{float(self.ols_df['beta_ci_high'].item()):.5f}]"
        )
        print(f"  Alpha:              {float(self.ols_df['alpha'].item()):.6f}")
        print(f"  Annualized Alpha    {float(self.ols_df['annualized_alpha'].item()):.3f}")
        print(f"  Alpha p-value       {float(self.ols_df['alpha_p_value'].item()):.4g}")
        print(f"  R-squared:          {float(self.ols_df['r_squared'].item()):.3f}")
        print(f"  Residual volatility: {float(self.ols_df['residual_volatility'].item()):.6f}")

        print(f"\nBeta significance")
        print(f"  Standard error:     {float(self.ols_df['beta_std_error'].item()):.4f}")
        print(f"  t-statistic:        {float(self.ols_df['beta_tstat'].item()):.2f}")
        print(f"  p-value:            {float(self.ols_df['beta_pvalue'].item()):.4g}")
        print("\n")





class MultiFactorRegression:
    """
    Generic OLS regression of one asset's returns against N other assets' returns
    at once. Not tied to any fixed factor set (e.g. Fama-French) — the
    "factors" here are just any other assets' return series the user supplies.
    """

    def __init__(self, asset1: pd.DataFrame, assets: dict[str, pd.DataFrame], return_type: str = "log"):

        if return_type not in {'log','simple'}:
            raise ValueError("return_type has to be either 'log' or 'simple'")


        col = f"{return_type}-returns"


        self.return_type = return_type

 
        if len(assets) < 1:
            raise ValueError("at least one factor asset is required")
        
        if not any(f"{col}" in col for col in asset1.columns):
            raise KeyError(f'No column named {col} in asset1 dataframe provided')

        if not any("date" in col for col in asset1.columns):
            raise KeyError(f'No date column in asset1 dataframe provided')
        

        y_df = asset1[["date", col]].copy()
        y_df = y_df.rename(columns = {col :'y'})

        
        merged = y_df


        assets_names = []

        for asset_name, asset_df in assets.items():
            
            if col not in asset_df.columns:
                raise KeyError(f"factor '{asset_name}' dataframe is missing column '{col}' — did you pass returns, not prices?")
            
            if 'date' not in asset_df.columns:
                raise KeyError(f"factor '{asset_name}' dataframe is missing column 'date'")
            
            
            df = asset_df[["date", col]].copy()
            
            df = df.rename(columns={col: asset_name.lower()})
            
            merged = pd.merge(merged, df, on="date", how="inner")
            
            assets_names.append(asset_name.lower())
            
        #inner-merging sequentially keeps only timestamps common to the dependent asset and every factor


        

        if merged.empty:
            raise ValueError("no overlapping timestamps across dependent asset and all factors")
 
        
        
        # merged.style.format({
        #     'date': '%Y-%m-%d'
        # })

        merged['date'] = pd.to_datetime(merged['date'])

        merged = merged.sort_values("date").reset_index(drop=True)

        merged = merged.dropna().reset_index(drop=True)

        
        
        self.x_col = [
            c for c in merged.columns
            if c not in {"date", "y"}
        ]


        self.y_series = merged["y"]
        self.x_series = merged[self.x_col]


        self.start_date = merged['date'].min()
        self.end_date = merged['date'].max()

        self.merged_df = merged.copy()

        


    def ols(self):

        y = self.y_series
        X = self.x_series

        X = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, X).fit()

        self.olsmodel = model

        return model

        
    

if __name__ == "__main__":
    pass