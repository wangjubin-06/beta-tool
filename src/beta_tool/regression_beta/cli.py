# regression_beta/cli.py
from .beta import Beta
from .multibeta import MultiBeta
from .portfoliobeta import PortfolioBeta

def beta(asset1: str, asset2: str, period: str = "1y"):
    result = Beta(asset1, asset2, period=period)
    # print result

def multibeta(assets: list[str], benchmark: str, period: str = "1y"):
    #...

def portfoliobeta(weights: str, benchmark: str):
    #...
