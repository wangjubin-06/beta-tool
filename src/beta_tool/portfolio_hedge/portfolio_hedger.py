from regression_beta.beta import Beta
from regression_beta.multibeta import MultiAssetsRegression
from regression_beta.portfoliobeta import PortfolioBeta
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta, datetime
import pandas as pd
import matplotlib.pyplot as plt
from data_collection.tiingo_api import TiingoApi
import os

class PortfolioHedge:

    ALLOWED_FREQUENCIES = {
        'daily',
        'weekly',
        'monthly'
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
            target: str | dict[str,float|int],
            hedge_instruments: str | list[str] = 'spy',
            backtest_start_date: str | None = None,
            backtest_end_date: str | None = None,
            backtest_period: str | None = None,
            frequency: str = 'daily',
            return_type: str = 'simple',
            hedge_type:str = 'static',    # "static" | "rolling"
            window: int | None = None,    # only used if hedge_type="rolling"
            rebalance_freq: int | None = None,
    ):

        if return_type not in self.RETURN_TYPES:

            raise ValueError("return_type = 'log' or 'simple' only.")

        
        if frequency not in self.ALLOWED_FREQUENCIES:

            raise ValueError("only daily, weekly, monthly is allowed for frequency!")


        if frequency == 'daily':
            print('daily data frequency may introduce market microstructure noise in the hedging')
        if frequency == 'weekly':
            print('weekly data frequency trades off data observation counts for less microstructure noise')
        if frequency == 'monthly':
            print('monthly data frequency may fail to capture changing market regimes in hedging')


        self.freq = frequency


        if hedge_type not in {'static','rolling'}:

            raise ValueError("hedge_type only can be 'static' or 'rolling'!")


        if hedge_type == 'rolling':

            self.rolling_window = self._rolling_window_resolver(window)

            self.rebalance_freq = self._rebalance_freq_resolver(rebalance_window=rebalance_freq)


    
        self.hedge_type = hedge_type

        
        self.return_type = return_type

        


        if not (isinstance(target,str) or isinstance(target,dict)):
            raise ValueError("target can only be a string of a single asset ticker or a dictionary mapping each asset ticker to its percentage in portfolio")
        
        self.target = target
        
        
        
        if not (isinstance(hedge_instruments, str) or isinstance(hedge_instruments, list)):
            raise TypeError("hedge instrument can only be a string of a single asset ticker or a list of hedge assets")

        self.hedge_instruments = hedge_instruments



        self._date_resolver(backtest_start_date=backtest_start_date, backtest_end_date=backtest_end_date, backtest_period=backtest_period)


        self.hedge_ratio_data = self._compute_hedge_ratio()



    # Public APIs

    def backtest(self, portfolio_starting_value: int|float = 1000000):

        # Do a comparison between unhedged position and hedged position with static and rolling beta

        if isinstance(portfolio_starting_value, int) or isinstance(portfolio_starting_value,float):
            if portfolio_starting_value > 0:
                self.port_starting_val = portfolio_starting_value
        else:
            raise ValueError('only numbers greater than 0 allowed for portfolio_starting_value')

        

        unhedged_port_val_series = self._port_val_series_unhedged()

        hedged_port_val_series = self._port_val_series_hedged()

        stats = self._port_stats(unhedged_port_val_series,hedged_port_val_series)


        self.unhedged_port_series = unhedged_port_val_series
        self.hedged_port_series = hedged_port_val_series


        self._stats_summary(stats)

        self.backtest_plot()

        return


    def backtest_plot(self):

        # Plots the price series of the hedged and unhedged portfolio over time

        ax1 = self._plot_price_series(self.unhedged_port_series)

        ax2 = self._plot_price_series(self.hedged_port_series)

        plt.show()
        


    # Private methods

    def _port_val_series_unhedged(self):

        if isinstance(self.target, dict):

            # Finding how much money is allocated to each asset at the start
            value_allocations = {}

            for ticker, percentage in self.target.items():
                value_allocations[ticker] = (percentage / 100) * self.port_starting_val


            # Finding the starting prices (price at backtest start date) for each asset in portfolio
            starting_prices = {}


            api_key = os.getenv('TIINGO_API_KEY')

            start_date = datetime.fromisoformat(self.backtest_start_date) - timedelta(days=2)
            end_date = datetime.fromisoformat(self.backtest_start_date) + timedelta(days=2)

            start_date = start_date.isoformat()
            end_date = end_date.isoformat()



            for ticker in self.target.keys():

                
                data_obj = TiingoApi(
                    api_key = api_key,
                    frequency = self.freq,
                    simplified = True
                )

                price_df = data_obj.get_data(
                    ticker=ticker,
                    start_date= start_date,
                    end_date = end_date
                )

                price_df = price_df[price_df["date"] >= pd.Timestamp(self.backtest_start_date)].copy()

                starting_price = float(price_df['adjClose'].iloc[0])

                starting_prices[ticker] = starting_price


            # Calculating the starting units of each asset
            starting_units = {}

            for ticker in self.target.keys():

                units = value_allocations[ticker] / starting_prices[ticker]

                starting_units[ticker] = units

            

            # Calculating the port value series of each asset in its own df
            price_series_dfs = {}

            for ticker in self.target.keys():

                data_obj = TiingoApi(
                    api_key = api_key,
                    frequency = self.freq,
                    simplified = True
                )

                price_df = data_obj.get_data(
                    ticker=ticker,
                    start_date= self.backtest_start_date,
                    end_date = self.backtest_end_date,
                )

                price_df.rename(columns={"adjClose": f"{ticker}-adjClose"}, inplace=True)

                price_df[f'{ticker}-port-price'] = price_df[f'{ticker}-adjClose'] * starting_units[ticker]

                price_df = price_df[['date',f'{ticker}-port-price']].copy()


                price_series_dfs[ticker] = price_df


            # Merging the dfs to get portfolio value series

            merged_df = list(price_series_dfs.values())[0]['date'].copy()

            for ticker in self.target.keys():

                merged_df = pd.merge(merged_df, price_series_dfs[ticker], on='date')

                cols = [f'{ticker}-port-price' for ticker in self.target.keys()]

                merged_df['port-val'] = merged_df[cols].sum(axis=1)



            portfolio_price_series_df = merged_df[['date','port-val']].copy()


        if isinstance(self.target, str):
            api_key = os.getenv('TIINGO_API_KEY')

            start_date = datetime.fromisoformat(self.backtest_start_date) - timedelta(days=2)
            end_date = datetime.fromisoformat(self.backtest_start_date) + timedelta(days=2)

            start_date = start_date.isoformat()
            end_date = end_date.isoformat()



            data_obj = TiingoApi(
                api_key = api_key,
                frequency = self.freq,
                simplified = True
            )

            price_df = data_obj.get_data(
                ticker= self.target,
                start_date= start_date,
                end_date = end_date
            )

            price_df = price_df[price_df["date"] >= pd.Timestamp(self.backtest_start_date)].copy()

            starting_price = float(price_df['adjClose'].iloc[0])


            units = self.port_starting_val / starting_price


            data_obj = TiingoApi(
                    api_key = api_key,
                    frequency = self.freq,
                    simplified = True
                )

            price_df = data_obj.get_data(
                ticker=ticker,
                start_date= self.backtest_start_date,
                end_date = self.backtest_end_date,
            )


            price_df[f'port-value'] = price_df['adjClose'] * units

            portfolio_price_series_df = price_df[['date','port-value']].copy()


        return portfolio_price_series_df



    def _compute_hedge_ratio(self):

        # --------- Hedge ratio data formats --------
        #
        # Static hedge:
        #     Single hedge instrument: returns a float number of the beta
        #     Multiple hedge instruments: returns a dictionary in the form {'ticker': beta}
        #
        # Rolling hedge:
        #     Single hedge instrument: returns a df with 'date' and 'rolling-beta' column, each date corresponds to rebalancing date
        #     Multiple hedge instruments: returns a df with 'date' and '{ticker}-rolling-beta' column, each date corresponds to rebalancing date
        #
        # Can also access latest betas by calling self.latest_beta


        # Hedging a single asset
        
        if isinstance(self.target,str):
            
            hedge_ratio_data = self._single_hedge()
            
        # Hedging a portfolio of assets with their weights
        
        elif isinstance(self.target,dict):
            
            hedge_ratio_data = self._multiple_hedge()

        return hedge_ratio_data
            
        

    # Hedging a single asset
    def _single_hedge(self):
        

        if isinstance(self.target,str):
            
            # single hedging instrument
            if isinstance(self.hedge_instruments,str):
            
                if self.hedge_type == 'rolling':

                    beta_obj = Beta(
                        asset1 = self.target,
                        asset2 = self.hedge_instruments,
                        start_date = self.data_start_date,
                        return_type = self.return_type,
                        frequency = self.freq,
                    )

                    beta_obj.historical_rolling_beta(window=self.rolling_window)

                    beta_data = beta_obj.get_rolling_beta()

                    self.latest_beta = float(beta_data.copy()['rolling_beta'].iloc[-1])
                    
                    # Today's beta is yesterday's one to prevent look-ahead bias
                    beta_data["rolling_beta"] = beta_data["rolling_beta"].shift(1)


                    beta_data = beta_data[beta_data["date"] >= pd.Timestamp(self.backtest_start_date)].copy()
                    beta_data = beta_data[beta_data["date"] <= pd.Timestamp(self.backtest_end_date)].copy()

                    beta_data = beta_data.reset_index(drop=True)

                    # Update the actual start and end of backtest to what is available in the dataframe with all merged by date
                    self.backtest_start_date = beta_data['date'].iloc[0].strftime('%Y-%m-%d')
                    self.backtest_end_date = beta_data['date'].iloc[-1].strftime('%Y-%m-%d')


                    # This dataframe gives the actual betas to use on each rebalance date
                    beta_readings = beta_data.iloc[::self.rebalance_freq][["date", "rolling_beta"]].copy()

                    beta_readings = beta_readings.reset_index(drop=True)


                    #print(beta_data.tail())
                    #print(beta_readings.tail())
                    
                else:

                    backtest_beta_obj = Beta(
                        asset1 = self.target,
                        asset2 = self.hedge_instruments,
                        end_date = self.backtest_start_date,
                        return_type = self.return_type,
                        frequency = self.freq,
                    )
                    
                    beta_data = backtest_beta_obj.get_static_beta()
                    

                    beta_readings = beta_data


                    full_beta_obj = Beta(
                        asset1= self.target,
                        asset2= self.hedge_instruments,
                        start_date = self.data_start_date,
                        return_type= self.return_type,
                        frequency = self.freq
                    )

                    self.latest_beta = full_beta_obj.get_static_beta()
                
            
            
            # multiple hedging instrument
            elif isinstance(self.hedge_instruments,list):

                if self.hedge_type == 'rolling':

                    beta_obj = MultiAssetsRegression(
                        asset1 = self.target,
                        assets = self.hedge_instruments,
                        frequency = self.freq,
                        start_date = self.data_start_date,
                        return_type = self.return_type,
                    )

                    beta_obj.historical_rolling_beta(window=self.rolling_window)

                    beta_data, independent_ticker_cols = beta_obj.get_rolling_beta()


                    latest_beta = {}

                    for col in independent_ticker_cols:
                        latest_beta[col] = float(beta_data.copy()[col].iloc[-1])

                    self.latest_beta = latest_beta


                    # Today's beta is yesterday's one to prevent look-ahead bias
                    beta_data[independent_ticker_cols] = beta_data[independent_ticker_cols].shift(1)


                    beta_data = beta_data[beta_data["date"] >= pd.Timestamp(self.backtest_start_date)].copy()
                    beta_data = beta_data[beta_data["date"] <= pd.Timestamp(self.backtest_end_date)].copy()

                    beta_data = beta_data.reset_index(drop=True)


                    # Update the actual start and end of backtest to what is available in the dataframe with all merged by date
                    self.backtest_start_date = beta_data['date'].iloc[0].strftime('%Y-%m-%d')
                    self.backtest_end_date = beta_data['date'].iloc[-1].strftime('%Y-%m-%d')


                    # This dataframe gives the actual betas to use on each rebalance date
                    beta_readings = beta_data.iloc[::self.rebalance_freq].copy()

                    beta_readings = beta_readings.reset_index(drop=True)


                    #print(beta_data.head())
                    #print(beta_readings.head())

                    
                else:

                    backtest_beta_obj = MultiAssetsRegression(
                        asset1 = self.target,
                        assets = self.hedge_instruments,
                        frequency = self.freq,
                        end_date = self.backtest_start_date,
                        return_type = self.return_type,
                    )
                    
                    beta_data = backtest_beta_obj.get_static_beta()

                    beta_readings = beta_data

                    full_beta_obj = MultiAssetsRegression(
                        asset1 = self.target,
                        assets = self.hedge_instruments,
                        frequency = self.freq,
                        start_date = self.data_start_date,
                        return_type = self.return_type,
                    )
                    self.latest_beta = full_beta_obj.get_static_beta()
                    
                    
            return beta_readings
        
        return None


    # Multiple hedge target
    def _multiple_hedge(self):
        
        if isinstance(self.target, dict):
            
            # Portfolio beta accepts single or multiple independent asset in regression

            if self.hedge_type == 'rolling':

                beta_obj = PortfolioBeta(
                    portfolio_dic = self.target,
                    asset_to_be_regressed = self.hedge_instruments,
                    frequency = self.freq,
                    start_date = self.data_start_date,
                    return_type = self.return_type
                )

                beta_obj.historical_rolling_beta(window=self.rolling_window)

                if isinstance(self.hedge_instruments,list):
                    beta_data, independent_ticker_cols = beta_obj.get_rolling_beta()

                    
                    latest_beta = {}

                    for col in independent_ticker_cols:
                        latest_beta[col] = float(beta_data.copy()[col].iloc[-1])
                    
                    self.latest_beta = latest_beta

                    # Today's beta is yesterday's one to prevent look-ahead bias
                    beta_data[independent_ticker_cols] = beta_data[independent_ticker_cols].shift(1)

                elif isinstance(self.hedge_instruments,str):
                    beta_data = beta_obj.get_rolling_beta()

                    self.latest_beta = float(beta_data.copy()['rolling-beta'].iloc[-1])

                    # Today's beta is yesterday's one to prevent look-ahead bias
                    beta_data['rolling-beta'] = beta_data['rolling-beta'].shift(1)


                beta_data = beta_data[beta_data["date"] >= pd.Timestamp(self.backtest_start_date)].copy()
                beta_data = beta_data[beta_data["date"] <= pd.Timestamp(self.backtest_end_date)].copy()

                beta_data = beta_data.reset_index(drop=True)

                # Update the actual start and end of backtest to what is available in the dataframe with all merged by date
                self.backtest_start_date = beta_data['date'].iloc[0].strftime('%Y-%m-%d')
                self.backtest_end_date = beta_data['date'].iloc[-1].strftime('%Y-%m-%d')


                # This dataframe gives the actual betas to use on each rebalance date
                beta_readings = beta_data.iloc[::self.rebalance_freq].copy()

                beta_readings = beta_readings.reset_index(drop=True)


                # print(beta_data.tail())
                # print(beta_readings.tail())
                # print(self.latest_beta)


                
            else:

                backtest_beta_obj = PortfolioBeta(
                    portfolio_dic = self.target,
                    asset_to_be_regressed = self.hedge_instruments,
                    frequency = self.freq,
                    end_date = self.backtest_start_date,
                    return_type = self.return_type
                )
                
                beta_data = backtest_beta_obj.get_static_beta()

                beta_readings = beta_data


                full_beta_obj = PortfolioBeta(
                    portfolio_dic = self.target,
                    asset_to_be_regressed = self.hedge_instruments,
                    frequency = self.freq,
                    start_date = self.data_start_date,
                    return_type = self.return_type
                )
                self.latest_beta = full_beta_obj.get_static_beta()

        
            return beta_readings

        return
        

    def _rolling_window_resolver(self, window:int | None):
        if window is None:
            return 35
        else:
            range_hi = 40
            range_lo = 30
            if window < range_lo or window > range_hi:
                print("Error: rolling window chosen is irregular for hedging purposes; reverted to default 35 observations.")
                return 35
            
            return window
            


    def _date_resolver(self, backtest_start_date, backtest_end_date, backtest_period):
        # Compute start date
        
        today = date.today()

        self.data_start_date = '1800-01-01'

        if backtest_period is not None:
            if backtest_period in self.ALLOWED_PERIODS:
                self.backtest_start_date = today - self.ALLOWED_PERIODS[backtest_period]
                self.backtest_end_date = today.isoformat()
            else:
                raise ValueError("backtest_period of the wrong format! choose from 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 20y, 30y.")
        else:
            if backtest_start_date is not None:
                self.backtest_start_date = date.fromisoformat(backtest_start_date)
            else:
                self.backtest_start_date = '1800-01-01'

            if backtest_end_date is not None:
                self.backtest_end_date = (date.fromisoformat(backtest_end_date) - timedelta(days=1)).isoformat()
            else:
                self.backtest_end_date = (today - timedelta(days=1)).isoformat()
        


    def _rebalance_freq_resolver(self, rebalance_window: int | None):

        floor = 30  # minimum observations for a statistically stable rolling regression

        if rebalance_window is None:
            return 126 if self.freq == 'daily' else 35
            # 126 is daily default;
            # keep 35 as placeholder default for weekly/monthly until validated

        if not isinstance(rebalance_window, int) or rebalance_window < floor:
            raise ValueError(
                f"window must be an integer >= {floor} observations "
                f"for a statistically reliable rolling regression; got {rebalance_window}."
            )

        return rebalance_window
        





if __name__ == '__main__':
    portfolio = PortfolioHedge(
        target= {'goog':40,'spy':60},
        hedge_instruments= ['ko','aapl'],
        backtest_start_date= '2024-09-09',
        backtest_end_date= '2025-09-09',
        frequency='daily',
        return_type='simple',
        hedge_type='rolling'
    )

    portfolio._compute_hedge_ratio()

