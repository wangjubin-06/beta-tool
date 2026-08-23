import pandas as pd
import numpy as np

from regression_beta.data import AssetData
from regression_beta.rolling import historical_rolling_beta, historical_rolling_beta_plot
from regression_beta.diagnostics import *
from regression_beta.plotting import two_asset_ols_plot

class PortfolioBeta:
    """
    
    This class takes a portfolio of assets and finds the asset weighted returns
    and regresses it against a benchmark to find portfolio beta
    
    """

    def __init__(self, portfolio_dic: dict[str: float]):
        """
        Parameters:
        portfolio_dic

        type: dictionary
        key-value pairs mapping asset ticker name to its percentage

        example:
            portfolio_dic = {
                "aapl": 10,
                "msft": 70,
                "nvda": 20
            }

        """

        # Initial checks
        if len(portfolio_dic) < 1:
            raise ValueError("portfolio cannot be empty")

        weight_sum = 0

        for weight in portfolio_dic.values():
            if not type(weight) == float or not type(weight) == int:
                raise ValueError("asset weightage has to be in numbers")
            if weight < 0 or weight > 100:
                raise ValueError("each asset weightage has to be between 0% to 100%")
            weight_sum += weight

        if float(weight_sum) != float(100):
            raise ValueError("sum of asset weights is not 100!")

        self.portfolio = portfolio_dic


    # Public API
    def regress(self):
        pass

