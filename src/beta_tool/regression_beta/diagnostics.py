import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

"""
    
This module has functions which test for heteroskedasticity
using Breusch-Pagan test, and test for autocorrelation using
Durbin-Watson / Ljung-Box, and also Variance Inflation Factor (multicollinearity, multi-factor only)

Heteroskedasticity and autocorrelation - 

Classical OLS gives you valid standard errors only if:
    -Residuals have constant variance (homoskedasticity)
    -Residuals aren't correlated with each other over time (no autocorrelation)


The variance of residuals directly shows how scattered the data points are away
from the best fit line.

In a return series regression, residuals may display heteroskedasticity (varying variance)
meaning the residuals' variance may change depending on 
the level of X (the independent variable); In the beta context: is the
regression more "noisy" during high-volatility periods than calm ones?
(Very likely yes for stock returns — this is called volatility clustering.)

if present, your beta_std_error, beta_pvalue, and confidence intervals
are biased — usually understated, making beta look more statistically significant than it is.


Residuals may also be autocorrelated
Autocorrelation asks whether consecutive residuals are correlated with each other.
i.e. does today's residual predict tomorrow's?

Daily return residuals are often mildly autocorrelated (momentum/mean-reversion
effects, or just imperfect timestamp alignment from your inner join).

Autocorrelated residuals again mean understated standard errors
— you'll think beta is more precisely estimated than it really is.
    

*note however that the actual beta regression tools can use HAC lags
to account for these two issues


Variance Inflation Factor (multicollinearity, multi-factor only): specific for multibeta.py

This is an issue present in a multi-factor beta regression.
It checks if your factors are correlated with each other

For example, a regression of TSLA against MSFT and AAPL
If AAPL and MSFT move almost identically, the regression
can't cleanly tell "how much of TSLA's move is explained by AAPL specifically" vs "by MSFT specifically"
— the coefficients become unstable and their standard errors blow up,
even though the overall fit (R²) might look fine.

    
"""

def heteroskedasticity(regresssion_model, alpha=0.05):
    """This uses the Breusch-Pagan Test in the statsmodels library
    to test for residual heteroskedasticity in the OLS regression model.

    Args:
        regresssion_model (statsmodels.regression.linear_model.OLSResults): the fitted results object
    """
    print('\n=========================================')
    print('Residual heteroskedasticity test results')
    print('=========================================')
            
    bp_test = het_breuschpagan(regresssion_model.resid, regresssion_model.model.exog)
    
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    
    for label, value in zip(labels, bp_test):
        print(f"{label}: {value:.4f}")
    
    
    # Interpretation
    p_value = bp_test[1]

    print("\nInterpretation:")
    if p_value < alpha:
        print(f"Heteroskedasticity detected (p-value < {alpha}).")
        print("   The residual variance is not constant.")
    else:
        print(f"No significant heteroskedasticity detected (p-value >= {alpha}).")
        print("   The residuals appear to have constant variance.")

    return
    
def autocorrelation(regression_model, alpha=0.05):
    """This uses the Durbin-Watson and Ljung-Box Test in the statsmodels library
    to test for residual autocorrelation in the OLS regression model.
    
    Durbin-Watson Interpretation:
    Value ~ 2: No autocorrelation
    Value < 2: Positive autocorrelation
    Value > 2: Negative autocorrelation.
    
    Ljung-Box Interpretation:
    lb_pvalue < significance level of 0.05: significant autocorrelation
    lb_pvalue > significance level of 0.05: no significant autocorrelation

    Args:
        regresssion_model (statsmodels.regression.linear_model.OLSResults): the fitted results object
    """
    print('\n=========================================')
    print('Residual autocorrelation test results')
    print('=========================================')
            
    residuals = regression_model.resid
    
    dw_stat = durbin_watson(residuals)
    
    print(f"\nDurbin-Watson statistic: {dw_stat:.4f}")
    
    print("\nDurbin-Watson Interpretation:")

    if dw_stat < 1.5:
        print("Possible positive autocorrelation.")
    elif dw_stat < 1.9:
        print("Some evidence of positive autocorrelation.")
    elif dw_stat <= 2.1:
        print("Little to no autocorrelation.")
    elif dw_stat <= 2.5:
        print("Some evidence of negative autocorrelation.")
    else:
        print("Possible negative autocorrelation.")

    
    lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    
    print("\nLjung-Box test results:")
    print(lb_result)
    
    # Get p-value for lag 10
    lb_pvalue = lb_result["lb_pvalue"].iloc[0]

    print("\nLjung-Box Interpretation:")

    if lb_pvalue < alpha:
        print(f"Significant autocorrelation detected (p-value < {alpha}).")
        print("   The residuals are not independent.")
    else:
        print(f"No significant autocorrelation detected (p-value >= {alpha}).")
        print("   There is insufficient evidence of autocorrelation.")
    
    return

def normality(regression_model, alpha=0.05):
    """Uses the Jarque-Bera test to test whether OLS residuals
    are normally distributed.

    H0: Residuals are normally distributed.
    H1: Residuals are not normally distributed.

    Args:
        regression_model: fitted statsmodels OLSResults object
        alpha: significance level, default = 0.05
    """
            
    residuals = regression_model.resid

    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)

    print("\n======================================")
    print("Residual normality test results")
    print("======================================")

    print(f"Jarque-Bera statistic: {jb_stat:.4f}")
    print(f"p-value              : {jb_pvalue:.4f}")
    print(f"Skewness             : {skew:.4f}")
    print(f"Kurtosis             : {kurtosis:.4f}")

    print("\nJarque-Bera Interpretation:")

    if jb_pvalue < alpha:
        print(f"Residuals are not normally distributed (p-value < {alpha}).")
        print("   There is significant evidence of non-normality.")
    else:
        print(f"Residuals appear normally distributed (p-value >= {alpha}).")
        print("   There is insufficient evidence of non-normality.")

    return


def multicollinearity(regression_model):
    """This uses the Variance Inflation Factor Test in the statsmodels library
    to test for asset returns collinearity in the OLS regression model.
    
    VIF Interpretation:
    VIF = 1: No correlation
    1 < VIF < 5: Moderate, manageable correlation
    VIF > 5: Severe multicollinearity

    Args:
        regresssion_model (statsmodels.regression.linear_model.OLSResults): the fitted results object
    """
    
    print('\n=========================================')
    print('Asset multicollinearity test results')
    print('=========================================')
    
    
    X = regression_model.model.exog 
    names = regression_model.model.exog_names

    # Calculate VIF for each feature
    vif_data = pd.DataFrame({
        "Variable": names,
        "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    })

    # Filter out the constant/intercept row, as VIF for it is not meaningful
    vif_data = vif_data[vif_data["Variable"] != "const"]
    print(vif_data)
    
    
    print("\nVIF Interpretation:")

    for _, row in vif_data.iterrows():
        variable = row["Variable"]
        vif = row["VIF"]

        if vif < 5:
            print(f"{variable}: VIF = {vif:.2f} — No serious multicollinearity.")
        elif vif < 10:
            print(f"{variable}: VIF = {vif:.2f} — Moderate multicollinearity.")
        else:
            print(f"{variable}: VIF = {vif:.2f} — Severe multicollinearity.")


