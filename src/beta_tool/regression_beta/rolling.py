import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# def historical_rolling_beta(y_df, x_df, y_col, x_col, return_type, window=60):
#     """
#     Initiate historical rolling OLS to get rolling beta and statistics
#     Function returns a dataframe with the data.

#     Parameters:
#         y_df: pandas dataframe
#             The time series of the dependent variable

#         x_df: pandas dataframe
#             The time series of the independent variable
        
#         y_col: string
#             The name of the data column for dependent variable dataframe

#         x_col: string
#             The name of the data column for independent variable dataframe

#         return_type: string
#             Either simple or log; denotes the returns type of both dataframes

#         window: integer (optional)

#         Specifies the look-back window for the rolling statistics
#         Defaults to 60 observations window
    
#     """

#     if window <= 0:
#         raise ValueError(
#             f"observation_window must be positive, got {window}."
#         )
    
#     if type(return_type) != str:
#         raise ValueError("return_type must be a string!")
#     if return_type not in {"simple","log"}:
#         raise ValueError("return_type can only either be 'simple' or 'log'!")
#     if not (type(x_col) == str or type(x_col) == list):
#         raise ValueError("x_col must be a string or list of column names!")
#     if type(y_col) != str:
#         raise ValueError("y_col must be a string!")
#     if type(window) != int:
#         raise ValueError("lookback window must be an integer!")
#     if window < 2:
#         raise ValueError("lookback window must be at least 2!")
    
#     if not "date" in y_df.columns:
#         raise KeyError("Missing date column in provided dependent variable dataframe")
#     if not "date" in x_df.columns:
#         raise KeyError("Missing date column in provided independent variable dataframe")
    
#     if not y_col in y_df.columns:
#         raise KeyError(f"Missing column named {y_col} in dependent variable dataframe!")

#     if type(x_col) == str:
#         if not x_col in x_df.columns:
#             raise KeyError(f"Missing column named {x_col} in independent variable dataframe!")

#     if type(x_col) == list:
#         if not pd.Series(x_col).isin(x_df.columns).all():
#             raise KeyError(f"Missing columns in independent variable dataframe!")


#     ydf = y_df.copy()
#     xdf = x_df.copy()


#     ydf['date'] = pd.to_datetime(ydf['date'], errors='coerce').dt.normalize()
#     xdf['date'] = pd.to_datetime(xdf['date'], errors='coerce').dt.normalize()

#     merged_df = pd.merge(
#         ydf,
#         xdf,
#         on = "date",
#         how = "inner"
#     ).sort_values('date')

#     if len(merged_df) < window:
#         raise ValueError(
#             f"Insufficient data for rolling beta: "
#             f"observation_window={window}, "
#             f"but only {len(merged_df)} observations are available."
#         )

#     y = merged_df[y_col]
#     x = merged_df[x_col]

#     x = sm.add_constant(x)

#     rols = RollingOLS(
#         endog=y,
#         exog=x,
#         window=window,
#     )

#     roll_results = rols.fit()


#     # Summarise results in a dataframe

#     # Convert RollingOLS outputs to NumPy arrays so that
#     # indexing is consistent regardless of the statsmodels
#     # return type.
#     params = np.asarray(roll_results.params)
#     bse = np.asarray(roll_results.bse)
#     tvalues = np.asarray(roll_results.tvalues)
#     pvalues = np.asarray(roll_results.pvalues)

#     # Column 0 = constant / alpha
#     # Column 1 = independent variable / beta
#     alpha = params[:, 0]
#     beta = params[:, 1]

#     beta_se = bse[:, 1]
#     beta_tstat = tvalues[:, 1]
#     beta_pvalue = pvalues[:, 1]


#     rolling_df = pd.DataFrame({
#         'date': merged_df['date'].to_numpy(),
#         'beta': beta,
#         'alpha': alpha,
#         'beta_se': beta_se,
#         'beta_tstat': beta_tstat,
#         'beta_pvalue': beta_pvalue,
#         'r_squared': np.asarray(roll_results.rsquared),
#         'residual_volatility': np.sqrt(np.asarray(roll_results.mse_resid)),
#     })

#     # 95% confidence interval for beta
#     rolling_df['beta_ci_lower'] = (
#         rolling_df['beta'] - 1.96 * rolling_df['beta_se']
#     )

#     rolling_df['beta_ci_upper'] = (
#         rolling_df['beta'] + 1.96 * rolling_df['beta_se']
#     )

#     # Annualized alpha
#     rolling_df['annualized_alpha'] = ((1 + rolling_df['alpha']) ** 252 - 1) if return_type == 'simple' else (rolling_df['alpha'] * 252)

#     # Remove observations before the first complete rolling window
#     rolling_df = rolling_df.dropna().reset_index(drop=True)


#     return rolling_df


# def rolling_beta_plot(rolling_df, window):
#     """
#     Visualise the rolling beta
#     """

#     fig, ax = plt.subplots(figsize=(12, 6))

#     ax.plot(
#         rolling_df['date'],
#         rolling_df['beta'],
#         label='Rolling Beta',
#         linewidth = 2,
#     )

#     ax.fill_between(
#         rolling_df['date'],
#         rolling_df['beta_ci_lower'],
#         rolling_df['beta_ci_upper'],
#         alpha=0.2,
#         label='95% Confidence Interval'
#     )

#     ax.axhline(
#         1.0,
#         color = 'black',
#         linestyle='--',
#         linewidth=1,
#         alpha = 0.4,
#         label='Beta = 1'
#     )

#     ax.axhline(
#         0.0,
#         linestyle=':',
#         linewidth=1
#     )

#     ax.set_title(f'{window}-observation Rolling Beta')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Beta')
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.show()

#     return fig


# def rolling_beta_summary(rolling_df, window):

#     latest = rolling_df.iloc[-1]
    
#     print("=" * 60)
#     print(f"Rolling Beta Summary ({window}-Observations Window)")
#     print("=" * 60)

#     print(f"\nObservation period")
#     print(f"  Start:              {rolling_df['date'].iloc[0].date()}")
#     print(f"  End:                {rolling_df['date'].iloc[-1].date()}")
#     print(f"  Observations:       {len(rolling_df)}")

#     print(f"\nCurrent estimates")
#     print(f"  Beta:               {float(latest['beta'].item()):.5f}")
#     print(
#         f"  95% CI:             "
#         f"[{float(latest['beta_ci_lower'].item()):.5f}, "
#         f"{float(latest['beta_ci_upper'].item()):.5f}]"
#     )
#     print(f"  Alpha:              {float(latest['alpha'].item()):.6f}")
#     print(f"  R-squared:          {float(latest['r_squared'].item()):.3f}")
#     print(f"  Residual volatility: {float(latest['residual_volatility'].item()):.6f}")

#     print(f"\nBeta significance")
#     print(f"  Standard error:     {float(latest['beta_se'].item()):.4f}")
#     print(f"  t-statistic:        {float(latest['beta_tstat'].item()):.2f}")
#     print(f"  p-value:            {float(latest['beta_pvalue'].item()):.4g}")

#     print(f"\nBeta history")
#     print(f"  Mean:               {float(rolling_df['beta'].mean()):.5f}")
#     print(f"  Median:             {float(rolling_df['beta'].median()):.5f}")
#     print(f"  Minimum:            {float(rolling_df['beta'].min()):.5f}")
#     print(f"  Maximum:            {float(rolling_df['beta'].max()):.5f}")
#     print(f"  Std. deviation:     {float(rolling_df['beta'].std()):.5f}")

#     print(f"\nR-squared history")
#     print(f"  Mean:               {float(rolling_df['r_squared'].mean()):.3f}")
#     print(f"  Minimum:            {float(rolling_df['r_squared'].min()):.3f}")
#     print(f"  Maximum:            {float(rolling_df['r_squared'].max()):.3f}")

#     print("=" * 60)

#     return


class MultiFactorRollingOLS:
    
    def __init__(self, merged_df: pd.DataFrame, y_col: str, x_col: list, return_type: str, window: int=60):
        
        if window <= 0:
            raise ValueError(
                f"observation_window must be positive, got {window}."
            )
        
        if return_type not in {'log','simple'}:
            raise ValueError(
                f"Wrong return type '{return_type}'. Only 'log' or 'simple' allowed."
            )
            
        if len(merged_df) < window:
            raise ValueError(
                f"Insufficient data for rolling beta: "
                f"observation_window={window}, "
                f"but only {len(merged_df)} observations are available."
            )
            
        if not y_col in merged_df.columns:
            raise ValueError(
                f"Missing column named '{y_col}' in merged_df provided."
            )
        
        if 'date' not in merged_df.columns:
            raise ValueError(
                f"Missing 'date' column in merged_df provided."
            )
            
        missing = [col for col in x_col if col not in merged_df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing} in merged_df provided.")
        
        
        merged_df = merged_df.dropna().reset_index(drop=True)
        
        
        
        self.merged_df = merged_df.copy()
        self.y_col = y_col
        self.x_col = x_col
        self.window = window
        self.return_type = return_type
        
        self.start_date = merged_df['date'].min()
        self.end_date = merged_df['date'].max()
        self.observations = len(merged_df)
        
    
    def rolling_ols(self):
        
        y = self.merged_df[self.y_col]
        X = self.merged_df[self.x_col]
        
        X = sm.add_constant(X, has_constant="add")
        
        
        window = self.window
        

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
                'alpha_std_error': bse['const'].values,
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
        print(f"Rolling Beta Summary ({self.window}-Observations Window)")
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
            print(f"  Alpha Standard Error:              {float(latest['alpha_std_error']):.6f}")
            print(f"  Alpha t-statistic:              {float(latest['alpha_tstat']):.6f}")
            print(f"  Alpha p-value:              {float(latest['alpha_pvalue']):.6f}")
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
        
        ax.set_title(f"{self.window}-observation rolling beta")
        ax.set_xlabel("Date")
        ax.set_ylabel("Beta")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.gcf().autofmt_xdate()  # Rotates dates automatically for readability
        ax.legend()

        plt.tight_layout()
        plt.show()

        return fig
    

class SingleFactorRollingOLS:
    
    """
    Initiate historical rolling OLS to get rolling beta and statistics
    Function returns a dataframe with the data.

    Parameters:
        y_df: pandas dataframe
            The time series of the dependent variable

        x_df: pandas dataframe
            The time series of the independent variable
        
        y_col: string
            The name of the data column for dependent variable dataframe

        x_col: string
            The name of the data column for independent variable dataframe

        return_type: string
            Either simple or log; denotes the returns type of both dataframes

        window: integer (optional)

        Specifies the look-back window for the rolling statistics
        Defaults to 60 observations window
    
    """
    
    def __init__(self, y_df: pd.DataFrame, y_col: str, x_df: pd.DataFrame, x_col: str, return_type:str, window: int = 60):
        
        if window <= 0:
            raise ValueError(
                f"observation_window must be positive, got {window}."
            )
    
        if type(return_type) != str:
            raise ValueError("return_type must be a string!")
        if return_type not in {"simple","log"}:
            raise ValueError("return_type can only either be 'simple' or 'log'!")
        if not (type(x_col) == str or type(x_col) == list):
            raise ValueError("x_col must be a string or list of column names!")
        if type(y_col) != str:
            raise ValueError("y_col must be a string!")
        if type(window) != int:
            raise ValueError("lookback window must be an integer!")
        if window < 2:
            raise ValueError("lookback window must be at least 2!")
        
        if not "date" in y_df.columns:
            raise KeyError("Missing date column in provided dependent variable dataframe")
        if not "date" in x_df.columns:
            raise KeyError("Missing date column in provided independent variable dataframe")
        
        if not y_col in y_df.columns:
            raise KeyError(f"Missing column named {y_col} in dependent variable dataframe!")

        if type(x_col) == str:
            if not x_col in x_df.columns:
                raise KeyError(f"Missing column named {x_col} in independent variable dataframe!")

        if type(x_col) == list:
            if not pd.Series(x_col).isin(x_df.columns).all():
                raise KeyError(f"Missing columns in independent variable dataframe!")
            
        
        ydf = y_df.copy()
        xdf = x_df.copy()


        ydf['date'] = pd.to_datetime(ydf['date'], errors='coerce').dt.normalize()
        xdf['date'] = pd.to_datetime(xdf['date'], errors='coerce').dt.normalize()

        merged_df = pd.merge(
            ydf,
            xdf,
            on = "date",
            how = "inner"
        ).sort_values('date')

        merged_df = merged_df.dropna().reset_index(drop=True)
        
        if len(merged_df) < window:
            raise ValueError(
                f"Insufficient data for rolling beta: "
                f"observation_window={window}, "
                f"but only {len(merged_df)} observations are available."
            )
        
        
        self.merged_df = merged_df.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.return_type = return_type
        self.window = window
        
        self.start_date = merged_df['date'].min()
        self.end_date = merged_df['date'].max()
        self.observations = len(merged_df)
        
        
    
    def rolling_ols(self):
        
        y = self.merged_df[self.y_col]
        x = self.merged_df[self.x_col]

        x = sm.add_constant(x)

        rols = RollingOLS(
            endog=y,
            exog=x,
            window=self.window,
        )

        roll_results = rols.fit()


        # Summarise results in a dataframe

        # Convert RollingOLS outputs to NumPy arrays so that
        # indexing is consistent regardless of the statsmodels
        # return type.
        params = np.asarray(roll_results.params)
        bse = np.asarray(roll_results.bse)
        tvalues = np.asarray(roll_results.tvalues)
        pvalues = np.asarray(roll_results.pvalues)

        # Column 0 = constant / alpha
        # Column 1 = independent variable / beta
        alpha = params[:, 0]
        beta = params[:, 1]

        beta_se = bse[:, 1]
        beta_tstat = tvalues[:, 1]
        beta_pvalue = pvalues[:, 1]


        rolling_df = pd.DataFrame({
            'date': self.merged_df['date'].to_numpy(),
            'beta': beta,
            'alpha': alpha,
            'beta_se': beta_se,
            'beta_tstat': beta_tstat,
            'beta_pvalue': beta_pvalue,
            'r_squared': np.asarray(roll_results.rsquared),
            'residual_volatility': np.sqrt(np.asarray(roll_results.mse_resid)),
        })

        # 95% confidence interval for beta
        rolling_df['beta_ci_lower'] = (
            rolling_df['beta'] - 1.96 * rolling_df['beta_se']
        )

        rolling_df['beta_ci_upper'] = (
            rolling_df['beta'] + 1.96 * rolling_df['beta_se']
        )

        # Annualized alpha
        rolling_df['annualized_alpha'] = ((1 + rolling_df['alpha']) ** 252 - 1) if self.return_type == 'simple' else (rolling_df['alpha'] * 252)

        # Remove observations before the first complete rolling window
        rolling_df = rolling_df.dropna().reset_index(drop=True)

        self.rolling_df = rolling_df
        
        return rolling_df
    
    
    def rolling_beta_plot(self):
        """
        Visualise the rolling beta
        """

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            self.rolling_df['date'],
            self.rolling_df['beta'],
            label='Rolling Beta',
            linewidth = 2,
        )

        ax.fill_between(
            self.rolling_df['date'],
            self.rolling_df['beta_ci_lower'],
            self.rolling_df['beta_ci_upper'],
            alpha=0.2,
            label='95% Confidence Interval'
        )

        ax.axhline(
            1.0,
            color = 'black',
            linestyle='--',
            linewidth=1,
            alpha = 0.4,
            label='Beta = 1'
        )

        ax.axhline(
            0.0,
            linestyle=':',
            linewidth=1
        )

        ax.set_title(f'{self.window}-observation Rolling Beta')
        ax.set_xlabel('Date')
        ax.set_ylabel('Beta')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig
    
    
    def rolling_beta_summary(self):
        latest = self.rolling_df.iloc[-1]
    
        print("=" * 60)
        print(f"Rolling Beta Summary ({self.window}-Observations Window)")
        print("=" * 60)

        print(f"\nObservation period")
        print(f"  Start:              {self.rolling_df['date'].iloc[0].date()}")
        print(f"  End:                {self.rolling_df['date'].iloc[-1].date()}")
        print(f"  Observations:       {len(self.rolling_df)}")

        print(f"\nCurrent estimates")
        print(f"  Beta:               {float(latest['beta'].item()):.5f}")
        print(
            f"  95% CI:             "
            f"[{float(latest['beta_ci_lower'].item()):.5f}, "
            f"{float(latest['beta_ci_upper'].item()):.5f}]"
        )
        print(f"  Alpha:              {float(latest['alpha'].item()):.6f}")
        print(f"  R-squared:          {float(latest['r_squared'].item()):.3f}")
        print(f"  Residual volatility: {float(latest['residual_volatility'].item()):.6f}")

        print(f"\nBeta significance")
        print(f"  Standard error:     {float(latest['beta_se'].item()):.4f}")
        print(f"  t-statistic:        {float(latest['beta_tstat'].item()):.2f}")
        print(f"  p-value:            {float(latest['beta_pvalue'].item()):.4g}")

        print(f"\nBeta history")
        print(f"  Mean:               {float(self.rolling_df['beta'].mean()):.5f}")
        print(f"  Median:             {float(self.rolling_df['beta'].median()):.5f}")
        print(f"  Minimum:            {float(self.rolling_df['beta'].min()):.5f}")
        print(f"  Maximum:            {float(self.rolling_df['beta'].max()):.5f}")
        print(f"  Std. deviation:     {float(self.rolling_df['beta'].std()):.5f}")

        print(f"\nR-squared history")
        print(f"  Mean:               {float(self.rolling_df['r_squared'].mean()):.3f}")
        print(f"  Minimum:            {float(self.rolling_df['r_squared'].min()):.3f}")
        print(f"  Maximum:            {float(self.rolling_df['r_squared'].max()):.3f}")

        print("=" * 60)

        return