import numpy as np
import pandas as pd
import statsmodels.api as sm
import regression_beta.returns as returns
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt

class OLSRegression:

    def __init__(self, asset1: pd.DataFrame, asset2: pd.DataFrame, asset_1_col, asset_2_col, frequency, return_type: str ="log", hac=False, hac_lags=None):
        
        if return_type not in {'log','simple'}:
            raise ValueError("return_type has to be either 'log' or 'simple'")


        self.hac = hac
        if self.hac:
            if type(hac_lags) == int and hac_lags > 0:
                self.hac_lags = hac_lags
            else:
                raise ValueError('error with hac and hac_lags configuration!')
        

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

        if self.hac is True:
            model = sm.OLS(self.y, x).fit(cov_type="HAC", cov_kwds={"maxlags": self.hac_lags})
        else:
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
                'Use HAC': self.hac,
                'HAC lags': (self.hac_lags if self.hac else 0),
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
        print(f"  Heteroskedasticity-Autocorrelation Robust Covariance:        {self.hac}")

        if self.hac:
            print(f"  Heteroskedasticity-Autocorrelation Robust Covariance lags:        {self.hac_lags}")

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


    def get_beta(self):
        return float(self.results.params[self.x_col])


    def rolling_ols(self, window=60):
        pass


class MultiFactorRegression:
    """
    Generic OLS regression of one asset's returns against N other assets' returns
    at once. Not tied to any fixed factor set (e.g. Fama-French) — the
    "factors" here are just any other assets' return series the user supplies.
    """

    def __init__(self, asset1: pd.DataFrame, assets: dict[str, pd.DataFrame], frequency, asset_1_col = None, return_type: str = "log", hac=False, hac_lags=None):

        if return_type not in {'log','simple'}:
            raise ValueError("return_type has to be either 'log' or 'simple'")

        self.hac = hac
        if self.hac:
            if type(hac_lags) == int and hac_lags > 0:
                self.hac_lags = hac_lags
            else:
                raise ValueError('error with hac and hac_lags configuration!')
        

        if asset_1_col is None:
            y_col = f"{return_type}-returns"
            x_col = f"{return_type}-returns"
        else:
            y_col = asset_1_col
            x_col = f"{return_type}-returns"
            


        self.return_type = return_type
        self.freq = frequency

 
        if len(assets) < 1:
            raise ValueError("at least one factor asset is required")
        
        if not any(f"{y_col}" in col for col in asset1.columns):
            raise KeyError(f'No column named {y_col} in asset1 dataframe provided')

        if not any("date" in col for col in asset1.columns):
            raise KeyError(f'No date column in asset1 dataframe provided')
        

        y_df = asset1[["date", y_col]].copy()
        y_df = y_df.rename(columns = {y_col :'y'})

        
        merged = y_df


        assets_names = []

        for asset_name, asset_df in assets.items():
            
            if x_col not in asset_df.columns:
                raise KeyError(f"factor '{asset_name}' dataframe is missing column '{x_col}' — did you pass returns, not prices?")
            
            if 'date' not in asset_df.columns:
                raise KeyError(f"factor '{asset_name}' dataframe is missing column 'date'")
            
            
            df = asset_df[["date", x_col]].copy()
            
            df = df.rename(columns={x_col: asset_name.lower()})
            
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

        if self.hac is True:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": self.hac_lags})
        else:
            model = sm.OLS(y,X).fit()


        self.olsmodel = model
        
        results = model
        # Store the stats as attributes
        conf_int = results.conf_int()
 
 
        self.alpha = float(results.params["const"])
        self.alpha_p_value = float(results.pvalues['const'])
        
        self.annualized_alpha = ((1 + float(results.params["const"])) ** 252 - 1) if self.return_type == 'simple' else (float(results.params["const"]) * 252)
        
        self.r_squared = float(results.rsquared)
        self.observations = int(results.nobs)
        self.residual_vol = float(results.resid.std())
 
        # per-asset stats, keyed by ticker, mirroring Beta's single-asset attributes
        self.betas = {}
        for name in self.x_col:
            self.betas[name] = {
                "beta": float(results.params[name]),
                "p_value": float(results.pvalues[name]),
                "t_stat": float(results.tvalues[name]),
                "std_error": float(results.bse[name]),
                "ci_low": float(conf_int.loc[name, 0]),
                "ci_high": float(conf_int.loc[name, 1]),
            }


        return model



    def summary(self, asset_1_name):
        
        print("=" * 60)
        print(f" Multi-Factor OLS Regression Summary")
        print(f"Portfolio consisting of \n{asset_1_name} against {', '.join(self.x_col)}")
        print("=" * 60)
        
        print(f"\nObservation period")
        print(f"  Start:              {str(self.start_date)}")
        print(f"  End:                {str(self.end_date)}")
        print(f"  Observations:       {self.observations}")
        print(f"  Frequency:          {self.freq}")
        print(f"  Return type:        {self.return_type}")
        print(f"  Heteroskedasticity-Autocorrelation Robust Covariance:        {self.hac}")
        
        if self.hac:
            print(f"  Heteroskedasticity-Autocorrelation Robust Covariance lags:        {self.hac_lags}")
            
        
        print(f"  Alpha:              {self.alpha:.6f}")
        print(f"  Annualized Alpha    {self.annualized_alpha:.3f}")
        print(f"  Alpha p-value       {self.alpha_p_value:.4g}")
        print(f"  R-squared:          {self.r_squared:.3f}")
        print(f"  Residual volatility: {self.residual_vol:.6f}")
                
                
        # lines = [
        #     '\n\n=====================================================',
        #     f"Multi-Factor OLS Regression: {asset_1_name} against {', '.join(self.x_col)}:",
        #     '=====================================================',
        #     f"{'Intercept':<30}: {self.intercept:.5f}",
        #     f"{'R-squared':<30}: {self.r_squared:.5f}",
        #     f"{'Start date':<30}: {self.start_date}",
        #     f"{'End date':<30}: {self.end_date}",
        #     f"{'Frequency':<30}: {self.freq}",
        #     f"{'No. of observations':<30}: {self.observations}",
        #     f"{'Use HAC':<30}: {self.hac}",
        #     f"{'Hac lags':<30}: {self.hac_lags if self.hac else 'None'}",
        #     f"{'Residual volatility':<30}: {self.residual_vol:.5f}",
        #     "",
        # ]
        
        print("=" * 60)
        print("Factors' betas")
        print("=" * 60)
        for name, stats in self.betas.items():
            print(f"{name.upper()}:  ")
            print(f"  Beta:               {stats['beta']:.5f}")
            print(
                f"  95% CI:             "
                f"[{stats['ci_low']:.5f}, "
                f"{stats['ci_high']:.5f}]"
            )
            
            print(f"\nBeta significance")
            print(f"  Standard error:     {stats['std_error']:.5f}")
            print(f"  t-statistic:        {stats['t_stat']:.5f}")
            print(f"  p-value:            {stats['p_value']:.4g}")
            print("-" * 60)
            print("\n")
            

    def rolling_ols(self, window=60):
        
        
        if window <= 0:
            raise ValueError(
                f"observation_window must be positive, got {window}."
            )

        if len(self.merged_df) < window:
            raise ValueError(
                f"Insufficient data for rolling beta: "
                f"observation_window={window}, "
                f"but only {len(self.merged_df)} observations are available."
            )
    
        y = self.y_series
        X = self.x_series.copy()

        X = sm.add_constant(X, has_constant="add")


        window = window
        self.rolling_window = window

        rols = RollingOLS(
            endog=y,
            exog=X, 
            window=window
        )

        results = rols.fit()

        df = self.merged_df
        x_col = self.x_col

        
        params = results.params

        self.beta_series = params

        # Convert the NumPy result arrays back into DataFrames
        bse = pd.DataFrame(
            results.bse,
            index=params.index,
            columns=params.columns
        )

        tvalues = pd.DataFrame(
            results.tvalues,
            index=params.index,
            columns=params.columns
        )

        pvalues = pd.DataFrame(
            results.pvalues,
            index=params.index,
            columns=params.columns
        )

        rsquared = pd.Series(
            results.rsquared,
            index=params.index
        )


        ci = results.conf_int(alpha=0.05)

        rolling_dfs = {}

        for ticker in x_col:

            rolling_df = pd.DataFrame({
                'date': df['date'].values,
                'beta': params[ticker].values,
                'beta_std_error': bse[ticker].values,
                'beta_tstat': tvalues[ticker].values,
                'beta_pvalue': pvalues[ticker].values,
                'alpha': params['const'].values,
                'alpha_stf_error': bse['const'].values,
                'alpha_tstat': tvalues['const'].values,
                'alpha_pvalue': pvalues['const'].values,
                'r_squared': rsquared.values
            })

            # Confidence intervals
            rolling_df["beta_ci_lower"] = (ci[(ticker, "lower")].values)
            rolling_df["beta_ci_upper"] = (ci[(ticker, "upper")].values)

            rolling_df["alpha_ci_lower"] = (ci[("const", "upper")].values)
            rolling_df["alpha_ci_upper"] = (ci[("const", "upper")].values)

            # Annualized alpha
            if self.return_type == "simple":
                rolling_df["annualized_alpha"] = (
                    (1 + rolling_df["alpha"]) ** 252 - 1
                )
            else:
                rolling_df["annualized_alpha"] = (
                    rolling_df["alpha"] * 252
                )

            rolling_dfs[ticker] = rolling_df



        self.rolling_dfs = rolling_dfs

        return rolling_dfs
    
        

        


    def rolling_beta_summary(self):
        print("=" * 60)
        print(f"Rolling Beta Summary ({self.rolling_window}-Observations Window)")
        print("=" * 60)

        print(f"\nObservation period")
        print(f"  Start:              {self.start_date}")
        print(f"  End:                {self.end_date}")
        print(f"  Observations:       {self.observations}")
        print("\n")

        for ticker, rolling_df in self.rolling_dfs.items():

            latest = rolling_df.iloc[-1]

            print("=" * 60)
            print(f"Stats for constituent independent variable {ticker}-returns")
            print("=" * 60)

            print(f"\nCurrent estimates")
            print(f"  Beta:               {float(latest['beta']):.5f}")
            print(
                f"  95% CI:             "
                f"[{float(latest['beta_ci_lower']):.5f}, "
                f"{float(latest['beta_ci_upper']):.5f}]"
            )
            print(f"  Alpha:              {float(latest['alpha']):.6f}")
            print(f"  R-squared:          {float(latest['r_squared']):.3f}")
            #print(f"  Residual volatility: {float(latest['residual_volatility'].item()):.6f}")

            print(f"\nBeta significance")
            print(f"  Standard error:     {float(latest['beta_std_error']):.4f}")
            print(f"  t-statistic:        {float(latest['beta_tstat']):.2f}")
            print(f"  p-value:            {float(latest['beta_pvalue']):.4g}")

            print(f"\nBeta history")
            print(f"  Mean:               {float(rolling_df['beta'].mean()):.5f}")
            print(f"  Median:             {float(rolling_df['beta'].median()):.5f}")
            print(f"  Minimum:            {float(rolling_df['beta'].min()):.5f}")
            print(f"  Maximum:            {float(rolling_df['beta'].max()):.5f}")
            print(f"  Std. deviation:     {float(rolling_df['beta'].std()):.5f}")

            print(f"\nR-squared history")
            print(f"  Mean:               {float(rolling_df['r_squared'].mean()):.3f}")
            print(f"  Minimum:            {float(rolling_df['r_squared'].min()):.3f}")
            print(f"  Maximum:            {float(rolling_df['r_squared'].max()):.3f}")

            #print("=" * 60)


    def rolling_beta_plot(self):
        
        fig, ax = plt.subplots(figsize=(12, 6))
        

        for ticker, df in self.rolling_dfs.items():

            df = df.copy()

            # Plot the beta trend line
            (line,) = ax.plot(
                df['date'],
                df['beta'],
                label = f'Rolling beta of {ticker}',
                linewidth = 1.6,
                alpha = 1,
            )

            # Plot the beta confidence intervals
            ax.fill_between(
                df['date'],
                df['beta_ci_lower'],
                df['beta_ci_upper'],
                color = line.get_color(),
                alpha = 0.2,
                label = f'{ticker} 95% Confidence interval'
            )

        ax.axhline(
            y=1,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.4,
            label="Beta = 1"
        )

        ax.axhline(
            y=0,
            color="black",
            linestyle=":",
            linewidth=1,
            alpha=0.4
        )
        
        ax.set_title(f"{self.rolling_window}-observation rolling beta")
        ax.set_xlabel("Date")
        ax.set_ylabel("Beta")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.gcf().autofmt_xdate()  # Rotates dates automatically for readability
        ax.legend()

        plt.tight_layout()
        plt.show()

        return fig
    
    
    
    
if __name__ == "__main__":
    pass