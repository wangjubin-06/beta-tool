# from .regression_beta import Beta, MultiBeta, PortfolioBeta
# from .portfolio_hedge import PortfolioHedge
# from .factor_research import EquityFactorsRegression


# __all__ = ["Beta", "MultiBeta", "PortfolioBeta", "PortfolioHedge", "EquityFactorsRegression"]


from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .regression_beta.beta import Beta
    from .regression_beta.multibeta import MultiBeta
    from .regression_beta.portfoliobeta import PortfolioBeta
    from .portfolio_hedge.portfolio_hedger import PortfolioHedge
    from .factor_research.factorsregression import EquityFactorsRegression
    

_EXPORTS = {
    "Beta": ".regression_beta.beta",
    "MultiBeta": ".regression_beta.multibeta",
    "PortfolioBeta": ".regression_beta.portfoliobeta",
    "PortfolioHedge": ".portfolio_hedge.portfolio_hedger",
    "EquityFactorsRegression": ".factor_research.factorsregression",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")