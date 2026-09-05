from regression_beta.beta import Beta
from regression_beta.multibeta import MultiAssetsRegression
from regression_beta.portfoliobeta import PortfolioBeta
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta
import pandas as pd

class PortfolioHedge:

    ALLOWED_FREQUENCIES = {
        'daily',
        'weekly',
        'monthly',
        'annually'
    }

    RETURN_TYPES = {
        'log',
        'simple'
    }

    ALLOWED_PERIODS = {
        "1m": relativedelta(months=1),
        "3m": relativedelta(months=3),
        "6m": relativedelta(months=6),
        "1y": relativedelta(years=1),
        "2y": relativedelta(years=2),
        "3y": relativedelta(years=3),
        "5y": relativedelta(years=5),
        "10y": relativedelta(years=10),
        "20y": relativedelta(years=20),
        "30y":relativedelta(years=30)
    }


    def __init__(
            self,
            target: str | dict[str,int],
            hedge_instruments: str | list[str] = 'spy',
            period: str = '1y',
            frequency: str = 'daily',
            return_type: str = 'log',
            hedge_type:str = 'static',    # "static" | "rolling"
            window: int = 126,    # only used if hedge_type="rolling"
            rebalance_freq: int | None = None,
    ):

        if return_type not in self.RETURN_TYPES:
            raise ValueError("return_type = 'log' or return_type = 'simple' only.")
        
        if frequency not in self.ALLOWED_FREQUENCIES:
            raise ValueError("only daily, weekly, monthly, annually is allowed for data interval!")

        if frequency != "daily":
            raise ValueError("PortfolioHedger currently supports interval='daily' only; window/rebalance_freq are calibrated in trading days.")

        if period not in self.ALLOWED_PERIODS:
            raise ValueError("'period' - choose from 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 20y, 30y")

        if hedge_type not in {'static','rolling'}:
            raise ValueError("hedge_type only can be 'static' or 'rolling'!")

        if hedge_type == 'rolling':

            self.hedge_type = hedge_type


            if window is None:
                raise ValueError("If hedge_type is 'rolling', please indicate window for the rolling Beta recalculation in the form of an integer!")
            if type(window) != int:
                raise TypeError("'window' has to be an integer!")
            if window < 40:
                raise ValueError("'window' rolling period is too low; beta estimates will be too uncertain!")


            self.rolling_window = window


            if rebalance_freq is None:
                raise ValueError("If hedge_type is 'rolling', please indicate 'frequency' in the number of days/months/years to recalculate hedge")


            self.rebalance_freq = rebalance_freq

        else:

            self.hedge_type = hedge_type
        

        self.return_type = return_type
        self.freq = frequency

        self.period = period

        

        if type(target) == str:
            self.regress_type = 'single'
        elif type(target) == dict:
            self.regress_type = 'portfolio'
        else:
            raise TypeError("target can only be a string of a single asset ticker or a dictionary mapping each asset ticker to its percentage in portfolio")
        
        if type(target) == str:
            self.multiple_targets = False
        elif type(target) == dict:
            self.multiple_targets = True
        else:
            raise ValueError("target can only be a string of a single asset ticker or a dictionary mapping each asset ticker to its percentage in portfolio")
        
        self.target = target
        

        if type(hedge_instruments) == str:
            self.hedge_instrument_count = 'single'
            self.hedge_instrument = hedge_instruments
        elif type(hedge_instruments) == list:
            self.hedge_instrument_count = 'many'
            self.hedge_instruments = hedge_instruments
        else:
            raise TypeError("hedge instrument can only be a string of a single asset ticker or a list of hedge assets")
        
        
        if type(hedge_instruments) == str:
            self.multiple_hedge = False
            self.hedge_instruments = hedge_instruments
        elif type(hedge_instruments) == list:
            self.multiple_hedge = True
            self.hedge_instruments = hedge_instruments
        else:
            raise TypeError("hedge instrument can only be a string of a single asset ticker or a list of hedge assets")



        # Compute start date
        if self.freq == "daily":
            
            today = date.today()
            
            start_date = today - self.ALLOWED_PERIODS[self.period]

            if self.hedge_type == 'rolling':
                # When fetching data for the hedger,
                # pull window extra trading days before
                # the period you actually want to analyze.
                # Then compute the rolling beta over that extended range.

                # Start date to be 'window' number of days before period.
                start_date = today - timedelta(days=self.rolling_window+7)


            self.start_date = start_date.isoformat()


    # Private methods

    def _compute_hedge_ratio(self):
        # dispatch to Beta / MultiBeta / PortfolioBeta based on
        # len(hedge_instruments) and whether target is a portfolio

        
        
        if self.hedge_type == 'rolling':

    
            if self.regress_type == 'single':

                target = self.hedge_instrument.lower()

                if self.hedge_instrument_count == 'single':

                    hedge_assets = self.hedge_instrument

                    beta_obj = Beta(
                        asset1=target,
                        asset2=hedge_assets,
                        start_date=start_date,
                        frequency=self.freq,
                        return_type=self.return_type,
                        hac=True
                    )

                    rolling_df = beta_obj.historical_rolling_beta(window = self.rolling_window)


                    # Rolling beta time series dataframe
                    rolling_df = rolling_df[['date','beta']].copy()
                    rolling_df['date'] = pd.to_datetime(rolling_df['date']).dt.normalize()
                    rolling_df = rolling_df.dropna().sort_values('date').reset_index(drop=True)
                    
                    
                    # Rebalance every N days, starting from start_date
                    rebalance_dates = pd.date_range(
                        start=start_date,
                        end=rolling_df['date'].iloc[-1],
                        freq=f'{self.rebalance_freq}D'
                    )
                    
                    # At each rebalance, use the most recent beta known BEFORE that date to prevent look-ahead bias
                    rebalance_beta = pd.merge_asof(
                        pd.DataFrame({'date': rebalance_dates}),
                        rolling_df[['date', 'beta']],
                        on='date',
                        direction='backward',
                        allow_exact_matches=False
                    )
                    
                    # For each date, use the most recent rebalance beta
                    final_beta_to_use_at_each_date = pd.merge_asof(
                        rolling_df.loc[rolling_df['date'] >= start_date, ['date']],
                        rebalance_beta,
                        on='date',
                        direction='backward'
                    )
                    
                    
                    



                else:
                    hedge_assets = self.hedge_instruments.copy()

                    beta_obj = MultiAssetsRegression(
                        asset1=target,
                        assets=hedge_assets,
                        period=self.period,
                        frequency=self.freq,
                        return_type=self.return_type,
                        hac=True
                    )

        
        # Single hedge target,
        
        if not self.multiple_targets:
            
            hedge_ratio = self._single_hedge()
            
        # Multiple hedge target:
        
        else:
            
            hedge_ratio = self._multiple_hedge()
            
        
        
        
    # Single hedge target
    def _single_hedge(self):
        
        if self.hedge_type == 'static':
            
            # single hedging instrument
            
            if not self.multiple_hedge:
                
                beta_obj = Beta(
                    asset1 = self.target,
                    asset2 = self.hedge_instruments,
                    start_date = self.start_date,
                    return_type = self.return_type,
                    frequency = self.freq,
                )
                
                return beta_obj.get_beta()
            
            # multiple hedging instrument
            else:
                
                beta_obj = MultiAssetsRegression(
                    asset1 = self.target,
                    assets = self.hedge_instruments,
                    frequency = self.freq,
                    start_date = self.start_date,
                    return_type = self.return_type,
                )
                
                return beta_obj.get_beta()
            
            
        else:
            pass
    
    
    # Multiple hedge target
    def _multiple_hedge(self):
        
        # single hedging instrument
        if not self.multiple_hedge:
            beta_obj = PortfolioBeta(
                portfolio_dic = self.target,
                asset_to_be_regressed = self.hedge_instruments,
                frequency = self.freq,
                start_date = self.start_date,
                return_type = self.return_type
            )
        
        # multiple hedging instrument
        else:
            pass
    
    
        

