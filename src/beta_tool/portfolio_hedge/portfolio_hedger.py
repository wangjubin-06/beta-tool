from regression_beta.beta import Beta
from regression_beta.multibeta import MultiAssetsRegression
from regression_beta.portfoliobeta import PortfolioBeta
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta, datetime
import pandas as pd
import matplotlib.pyplot as plt
from regression_beta.data import AssetData
from regression_beta.returns import simple_returns
from portfolio_hedge.metrics import *


class PortfolioHedge:

    ALLOWED_FREQUENCIES = {
        'daily',
        'weekly',
        'monthly'
    }

    RETURN_TYPES = {
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

    WINDOW_DEFAULTS = {'daily': 126, 'weekly': 52, 'monthly': 24}
    WINDOW_FLOORS   = {'daily': 30,  'weekly': 20, 'monthly': 12}

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

            raise ValueError("return_type = 'simple' only.")

        
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

            self.rebalance_freq = self._rebalance_freq_resolver(rebalance_freq=rebalance_freq)


    
        self.hedge_type = hedge_type

        
        self.return_type = return_type

        


        if not (isinstance(target,str) or isinstance(target,dict)):
            raise ValueError("target can only be a string of a single asset ticker or a dictionary mapping each asset ticker to its percentage in portfolio")
        
        self.target = target
        
        
        
        if not (isinstance(hedge_instruments, str) or isinstance(hedge_instruments, list)):
            raise TypeError("hedge instrument can only be a string of a single asset ticker or a list of hedge assets")

        if isinstance(hedge_instruments,list):
            lst = [ticker.lower() for ticker in hedge_instruments]

            self.hedge_instruments = lst
        else:
            self.hedge_instruments = hedge_instruments.lower()



        self._date_resolver(backtest_start_date=backtest_start_date, backtest_end_date=backtest_end_date, backtest_period=backtest_period)

        self.effective_backtest_start_date = self.backtest_start_date
        self.effective_backtest_end_date = self.backtest_end_date





    # Public APIs

    def backtest(self):

        # Do a comparison between unhedged position and hedged position with static and rolling beta
        

        if self.hedge_type == 'rolling':

            self._compute_hedge_ratio()   # resolves effective_backtest_start/end_date up front


        self._port_return_series()

        unhedged_port_return_series = self.unhedged_port_return_series


        if self.hedge_type == 'rolling':

            self._rolling_hedged_return_series()

            hedged_port_return_series = self.rolling_hedged_return_series

        elif self.hedge_type == 'static':

            self._static_hedged_return_series()

            hedged_port_return_series = self.static_hedged_return_series



        metrics_df, hedge_df = calculate_risk_metrics(
            hedged_port_return_series,
            return_col="port-returns",
            hedged_col="hedged-returns",
            date_col="date",
            frequency="daily",
            risk_free_rate=0.0,
            plot=True,
        )

        print(metrics_df)
        print('\n\n')
        print(hedge_df)


        # FOR WEEKLY
        # metrics_df, hedge_df = calculate_risk_metrics(
        #     hedged_port_return_series,
        #     frequency="weekly",
        #     risk_free_rate=0.04,
        # )


        # FOR MONTHLY 
        # metrics_df, hedge_df = calculate_risk_metrics(
        #     hedged_port_return_series,
        #     frequency="monthly",
        #     risk_free_rate=0.04,
        # )

        # IF WANT TO PLOT
        # metrics_df, hedge_df = calculate_risk_metrics(
        #     hedged_port_return_series,
        #     frequency="daily",
        #     risk_free_rate=0.04,
        #     plot=True,
        # )



    def backtest_plot(self):

        self._plot()










    # Private methods


    # =================================================
    # Unhedged portfolio value series and return series
    # =================================================

    # Returns a df with column named 'date' and 'port-value' to self.unhedged_portfolio_price_series
    def _port_val_series_unhedged(self, portfolio_starting_value: int|float = 100):

        if not (isinstance(portfolio_starting_value, int) or isinstance(portfolio_starting_value, float)):
            raise ValueError('wrong input type for portfolio_starting_value')
        if not portfolio_starting_value > 0:
            raise ValueError('only numbers greater than 0 allowed for portfolio_starting_value')


        
        start_date = datetime.fromisoformat(self.effective_backtest_start_date) - timedelta(days=2)
        end_date = datetime.fromisoformat(self.effective_backtest_end_date) + timedelta(days=2)

        start_date = start_date.isoformat()
        end_date = end_date.isoformat()

        if isinstance(self.target, dict):

            # Finding how much money is allocated to each asset at the start
            value_allocations = {}

            for ticker, percentage in self.target.items():
                value_allocations[ticker] = (percentage / 100) * portfolio_starting_value


            # Finding the starting prices (price at backtest start date) for each asset in portfolio
            starting_prices = {}


            for ticker in self.target.keys():

                data_obj = AssetData(
                    ticker = ticker,
                    frequency = self.freq,
                    start_date = start_date,
                    end_date = end_date
                )

                price_df = data_obj.get_prices()


                price_df = price_df[price_df["date"] >= pd.Timestamp(self.effective_backtest_start_date)].copy()

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

                data_obj = AssetData(
                    ticker = ticker,
                    frequency=self.freq,
                    start_date= self.effective_backtest_start_date,
                    end_date= self.effective_backtest_end_date
                )

                price_df = data_obj.get_prices()



                price_df.rename(columns={"adjClose": f"{ticker}-adjClose"}, inplace=True)

                price_df[f'{ticker}-port-value'] = price_df[f'{ticker}-adjClose'] * starting_units[ticker]

                price_df = price_df[['date',f'{ticker}-port-value']].copy()


                price_series_dfs[ticker] = price_df


            # Merging the dfs to get portfolio value series
            merged_df = None

            for ticker, df in price_series_dfs.items():

                if merged_df is None:
                    merged_df = df.copy()
                else:
                    merged_df = pd.merge(
                        merged_df,
                        df,
                        on='date',
                        how='inner'
                    )

            cols = [f'{ticker}-port-value' for ticker in self.target.keys()]

            merged_df['port-value'] = merged_df[cols].sum(axis=1)



            portfolio_price_series_df = merged_df[['date','port-value']].copy()


        if isinstance(self.target, str):


            data_obj = AssetData(
                ticker = self.target,
                frequency=self.freq,
                start_date= start_date,
                end_date = end_date
            )

            price_df = data_obj.get_prices()


            price_df = price_df[price_df["date"] >= pd.Timestamp(self.effective_backtest_start_date)].copy()

            starting_price = float(price_df['adjClose'].iloc[0])


            units = portfolio_starting_value / starting_price


            data_obj = AssetData(
                ticker = self.target,
                frequency=self.freq,
                start_date= self.effective_backtest_start_date,
                end_date = self.effective_backtest_end_date
            )

            price_df = data_obj.get_prices()


            price_df[f'port-value'] = price_df['adjClose'] * units

            portfolio_price_series_df = price_df[['date','port-value']].copy()


        self.unhedged_portfolio_price_series = portfolio_price_series_df.copy()

    # Returns a df with column named 'date' and 'port-returns' to self.unhedged_port_return_series
    def _port_return_series(self, portfolio_starting_value: int | float = 100):

        self._port_val_series_unhedged(portfolio_starting_value=portfolio_starting_value)

        price_df = self.unhedged_portfolio_price_series.copy()
        price_df = price_df.sort_values('date').reset_index(drop=True)

        price_df['port-returns'] = price_df['port-value'].pct_change()
        price_df = price_df.dropna(subset=['port-returns']).reset_index(drop=True)

        self.unhedged_port_return_series = price_df[['date', 'port-returns']].copy()

        # start_date = datetime.fromisoformat(self.backtest_start_date) - timedelta(days=5)
        # end_date = datetime.fromisoformat(self.backtest_end_date) + timedelta(days=2)

        # start_date = start_date.isoformat()
        # end_date = end_date.isoformat()


        # if isinstance(self.target,str):

        #     data_obj = AssetData(
        #         ticker=self.target,
        #         frequency=self.freq,
        #         start_date = start_date,
        #         end_date = end_date
        #     )

        #     price_df = data_obj.get_prices()

        #     returns_df = simple_returns(data = price_df)

        #     returns_df = returns_df[returns_df["date"] >= pd.Timestamp(self.backtest_start_date)].copy()
        #     returns_df = returns_df[returns_df["date"] <= pd.Timestamp(self.backtest_end_date)].copy()

        #     returns_df.rename(columns={"simple-returns": "port-returns"}, inplace=True)

        #     self.unhedged_port_return_series = returns_df.copy()

        #     #return returns_df


        # elif isinstance(self.target,dict):

        #     returns_dict = {}
        #     cols = []

        #     for ticker, allocation in self.target.items():

        #         data_obj = AssetData(
        #             ticker= ticker,
        #             frequency=self.freq,
        #             start_date = start_date,
        #             end_date = end_date
        #         )

        #         price_df = data_obj.get_prices()

        #         returns_df = simple_returns(data = price_df)

        #         returns_df = returns_df[returns_df["date"] >= pd.Timestamp(self.backtest_start_date)].copy()
        #         returns_df = returns_df[returns_df["date"] <= pd.Timestamp(self.backtest_end_date)].copy()

        #         returns_df[f'{ticker}-port-returns'] = returns_df['simple-returns'] * (allocation / 100)

        #         returns_df = returns_df[['date',f'{ticker}-port-returns']].copy()

        #         returns_dict[ticker] = returns_df

        #         cols.append(f'{ticker}-port-returns')


        #     merged_df = None

        #     for ticker, df in returns_dict.items():

        #         if merged_df is None:
        #             merged_df = df.copy()
        #         else:
        #             merged_df = pd.merge(
        #                 merged_df,
        #                 df,
        #                 on='date',
        #                 how='inner'
        #             )

            

        #     merged_df['port-returns'] = merged_df[cols].sum(axis=1)



        #     port_return_series = merged_df[['date','port-returns']].copy()


        #     self.unhedged_port_return_series = port_return_series.copy()




    # =================================================
    # Hedge instruments return series
    # =================================================

    # Either returns a single df with column named 'date' and 'hedge-returns' OR a tuple(df,list) of column names for the hedge instrument column names
    def _hedge_instrument_return_series(self):

        start_date = datetime.fromisoformat(self.effective_backtest_start_date) - timedelta(days=5)
        end_date = datetime.fromisoformat(self.effective_backtest_end_date) + timedelta(days=2)

        start_date = start_date.isoformat()
        end_date = end_date.isoformat()


        # Returns a single df with column named 'hedge-returns'
        if isinstance(self.hedge_instruments, str):

            data_obj = AssetData(
                ticker=self.hedge_instruments,
                frequency=self.freq,
                start_date = start_date,
                end_date = end_date
            )

            price_df = data_obj.get_prices()

            returns_df = simple_returns(data = price_df)

            returns_df = returns_df[returns_df["date"] >= pd.Timestamp(self.effective_backtest_start_date)].copy()
            returns_df = returns_df[returns_df["date"] <= pd.Timestamp(self.effective_backtest_end_date)].copy()

            returns_df.rename(columns={"simple-returns": "hedge-returns"}, inplace=True)


            return returns_df


        # Returns a tuple(df,list) of column names for the hedge instrument column names
        elif isinstance(self.hedge_instruments, list):

            returns_dict = {}
            cols = []

            for ticker in self.hedge_instruments:

                data_obj = AssetData(
                    ticker= ticker,
                    frequency=self.freq,
                    start_date = start_date,
                    end_date = end_date
                )

                price_df = data_obj.get_prices()

                returns_df = simple_returns(data = price_df)

                returns_df = returns_df[returns_df["date"] >= pd.Timestamp(self.effective_backtest_start_date)].copy()
                returns_df = returns_df[returns_df["date"] <= pd.Timestamp(self.effective_backtest_end_date)].copy()

                returns_df.rename(columns={"simple-returns": f"{ticker}-hedge-returns"}, inplace=True)


                returns_df = returns_df[['date',f'{ticker}-hedge-returns']].copy()

                returns_dict[ticker] = returns_df

                cols.append(f'{ticker}-hedge-returns')


            merged_df = None

            for ticker, df in returns_dict.items():

                if merged_df is None:
                    merged_df = df.copy()
                else:
                    merged_df = pd.merge(
                        merged_df,
                        df,
                        on='date',
                        how='inner'
                    )


            hedge_return_series = merged_df.copy()


            return hedge_return_series, cols





    # =================================================
    # Hedged portfolio return series
    # =================================================

    # returns a df with 'date' and 'hedged-returns' to self.static_hedged_return_series
    def _static_hedged_return_series(self):

        port_returns_series = self.unhedged_port_return_series


        if isinstance(self.hedge_instruments, str):

            hedge_instrument_return_series = self._hedge_instrument_return_series()

            merged_df = pd.merge(
                port_returns_series,
                hedge_instrument_return_series,
                on='date',
                how='inner'
            )

            static_beta_float = self._compute_hedge_ratio()

            merged_df['adjusted-hedge-returns'] = merged_df['hedge-returns'] * static_beta_float

            merged_df['hedged-returns'] = merged_df['port-returns'] - merged_df['adjusted-hedge-returns']

            #df = merged_df[['date','hedged-returns']].copy()

            self.static_hedged_return_series = merged_df.copy()


        elif isinstance(self.hedge_instruments, list):


            hedge_instrument_return_series, cols = self._hedge_instrument_return_series()

            merged_df = pd.merge(
                port_returns_series,
                hedge_instrument_return_series,
                on='date',
                how='inner'
            )



            static_beta_dict = self._compute_hedge_ratio()


            sum_cols = []


            for col in cols:

                ticker = col[:-14]


                merged_df[f'adjusted-{ticker}-hedge-returns'] = merged_df[col] * static_beta_dict[ticker]


                sum_cols.append(f'adjusted-{ticker}-hedge-returns')



            merged_df['total-hedge-returns'] = merged_df[sum_cols].sum(axis=1)


            merged_df['hedged-returns'] = merged_df['port-returns'] - merged_df['total-hedge-returns']

            #df = merged_df[['date','hedged-returns']].copy()


            self.static_hedged_return_series = merged_df.copy()


    # returns a df with 'date' and 'hedged-returns' to self.rolling_hedged_return_series
    def _rolling_hedged_return_series(self):

        
        port_returns_series = self.unhedged_port_return_series

        if isinstance(self.hedge_instruments, str):


            hedge_instrument_return_series = self._hedge_instrument_return_series()


            merged_df = pd.merge(
                port_returns_series,
                hedge_instrument_return_series,
                on='date',
                how='inner'
            )

            

            rolling_beta_float = self._compute_hedge_ratio()

            
            merged_df = merged_df.sort_values('date')
            rolling_beta_float = rolling_beta_float.sort_values('date')

            final_df = pd.merge_asof(
                merged_df,
                rolling_beta_float,
                on='date',
                direction='backward'
            )

            
            
            final_df['adjusted-hedge-returns'] = final_df['rolling-beta'] * final_df['hedge-returns']

            final_df['hedged-returns'] = final_df['port-returns'] - final_df['adjusted-hedge-returns']

            #df = final_df[['date','hedged-returns']].copy()

            self.rolling_hedged_return_series = final_df.copy()

            


        if isinstance(self.hedge_instruments, list):


            hedge_instrument_return_series, cols = self._hedge_instrument_return_series()


            merged_df = pd.merge(
                port_returns_series,
                hedge_instrument_return_series,
                on='date',
                how='inner'
            )


            rolling_beta_df = self._compute_hedge_ratio()


            merged_df = merged_df.sort_values('date')

            rolling_beta_df = rolling_beta_df.sort_values('date')

            final_df = pd.merge_asof(
                merged_df,
                rolling_beta_df,
                on='date',
                direction='backward'
            )


            ticker_list = []
            for col in cols:
                ticker = col[:-14]
                ticker_list.append(ticker)


            sum_cols = []

            for ticker in ticker_list:
                final_df[f'{ticker}-adjusted-hedge-returns'] = final_df[f'{ticker}-rolling-beta'] * final_df[f'{ticker}-hedge-returns']

                sum_cols.append(f'{ticker}-adjusted-hedge-returns')


            final_df['total-hedge-returns'] = final_df[sum_cols].sum(axis=1)


            final_df['hedged-returns'] = final_df['port-returns'] - final_df['total-hedge-returns']

            #df = final_df[['date','hedged-returns']].copy()


            self.rolling_hedged_return_series = final_df.copy()





    # =================================================
    # Hedged portfolio price series
    # =================================================

    # Returns a df with column named 'date' and 'port-value' to self.hedged_portfolio_price_series
    def _port_val_series_hedged(self, portfolio_starting_value: int|float = 100):

        if not (isinstance(portfolio_starting_value, int) or isinstance(portfolio_starting_value, float)):
            raise ValueError('wrong input type for portfolio_starting_value')
        if not portfolio_starting_value > 0:
            raise ValueError('only numbers greater than 0 allowed for portfolio_starting_value')


        if not hasattr(self, 'unhedged_port_return_series'):
            self._port_return_series(portfolio_starting_value=portfolio_starting_value)


        if self.hedge_type == "static":
            if not hasattr(self, 'static_hedged_return_series'):
                self._static_hedged_return_series()
            df = self.static_hedged_return_series.copy()

        elif self.hedge_type == "rolling":
            if not hasattr(self, 'rolling_hedged_return_series'):
                self._rolling_hedged_return_series()
            df = self.rolling_hedged_return_series.copy()

        df['port-value'] = portfolio_starting_value * (1 + df['hedged-returns']).cumprod()

        self.hedged_portfolio_price_series = df.copy()


        # if not (isinstance(portfolio_starting_value, int) or isinstance(portfolio_starting_value, float)):
        #     raise ValueError('wrong input type for portfolio_starting_value')
        # if not portfolio_starting_value > 0:
        #     raise ValueError('only numbers greater than 0 allowed for portfolio_starting_value')



        # if self.hedge_type == "static":

        #     df = self.static_hedged_return_series.copy()

        #     df['port-value'] = portfolio_starting_value * (1 + df['hedged-returns']).cumprod()

        #     #df = df[['date','port-value']].copy()



        # elif self.hedge_type == "rolling":

        #     df = self.rolling_hedged_return_series.copy()

        #     df['port-value'] = portfolio_starting_value * (1 + df['hedged-returns']).cumprod()

        #     #df = df[['date','port-value']].copy()

            
        
        # self.hedged_portfolio_price_series = df.copy()
        


    # =================================================
    # Calculate the hedge ratios depending on static or rolling
    # =================================================

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


        if hasattr(self, '_cached_hedge_ratio'):
            return self._cached_hedge_ratio

        
        # Hedging a single asset
        if isinstance(self.target,str):
            
            hedge_ratio_data = self._single_hedge()
            
        # Hedging a portfolio of assets with their weights
        
        elif isinstance(self.target,dict):
            
            hedge_ratio_data = self._multiple_hedge()

        else:
            hedge_ratio_data = None


        self._cached_hedge_ratio = hedge_ratio_data

        return hedge_ratio_data
            

    # Hedging a single asset
    def _single_hedge(self):
        

        if isinstance(self.target,str):
            
            # single hedging instrument
            if isinstance(self.hedge_instruments,str):

                # Returns a df with one column 'rolling-beta'
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

                    self.rolling_beta_df = beta_data.copy()
                    self.rolling_beta_df_cols = ['rolling_beta']

                    self.latest_beta = float(beta_data.copy()['rolling_beta'].iloc[-1])
                    
                    # Today's beta is yesterday's one to prevent look-ahead bias
                    beta_data["rolling_beta"] = beta_data["rolling_beta"].shift(1)


                    # Drop the first observation since it will be NaN after shifting by 1
                    beta_data.dropna(ignore_index=True, inplace=True)


                    beta_data = beta_data[beta_data["date"] >= pd.Timestamp(self.backtest_start_date)].copy()
                    beta_data = beta_data[beta_data["date"] <= pd.Timestamp(self.backtest_end_date)].copy()
                

                    # Update the actual start and end of backtest to what is available in the dataframe with all merged by date
                    self.effective_backtest_start_date = beta_data['date'].iloc[0].strftime('%Y-%m-%d')
                    self.effective_backtest_end_date = beta_data['date'].iloc[-1].strftime('%Y-%m-%d')

                    beta_data.rename(columns={"rolling_beta": "rolling-beta"}, inplace=True)

                    # This dataframe gives the actual betas to use on each rebalance date
                    beta_readings = beta_data.iloc[::self.rebalance_freq][["date", "rolling-beta"]].copy()

                    beta_readings = beta_readings.reset_index(drop=True)


                    #print(beta_data.tail())
                    #print(beta_readings.tail())
                

                # Returns a floating point number
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

                # Returns a df with many columns '{ticker}-rolling-beta'
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

                    self.rolling_beta_df = beta_data.copy()
                    self.rolling_beta_df_cols = independent_ticker_cols.copy()


                    latest_beta = {}

                    for col in independent_ticker_cols:
                        latest_beta[col] = float(beta_data.copy()[col].iloc[-1])

                    self.latest_beta = latest_beta


                    # Today's beta is yesterday's one to prevent look-ahead bias

                    for col in independent_ticker_cols:
                        beta_data[col] = beta_data[col].shift(1)

                    # Drop the first observation since it will be NaN after shifting by 1
                    beta_data.dropna(ignore_index=True, inplace=True)


                    beta_data = beta_data[beta_data["date"] >= pd.Timestamp(self.backtest_start_date)].copy()
                    beta_data = beta_data[beta_data["date"] <= pd.Timestamp(self.backtest_end_date)].copy()


                    # Update the actual start and end of backtest to what is available in the dataframe with all merged by date
                    self.effective_backtest_start_date = beta_data['date'].iloc[0].strftime('%Y-%m-%d')
                    self.effective_backtest_end_date = beta_data['date'].iloc[-1].strftime('%Y-%m-%d')


                    # This dataframe gives the actual betas to use on each rebalance date
                    beta_readings = beta_data.iloc[::self.rebalance_freq].copy()

                    beta_readings = beta_readings.reset_index(drop=True)


                    #print(beta_data.head())
                    #print(beta_readings.head())


                # Returns a dict of {ticker}: {beta}
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


                # Returns a df with many columns '{ticker}-rolling-beta'
                if isinstance(self.hedge_instruments,list):
                    beta_data, independent_ticker_cols = beta_obj.get_rolling_beta()

                    self.rolling_beta_df = beta_data.copy()
                    self.rolling_beta_df_cols = independent_ticker_cols.copy()

                    latest_beta = {}

                    for col in independent_ticker_cols:
                        latest_beta[col] = float(beta_data.copy()[col].iloc[-1])
                    
                    self.latest_beta = latest_beta

                    # Today's beta is yesterday's one to prevent look-ahead bias


                    for col in independent_ticker_cols:
                        beta_data[col] = beta_data[col].shift(1)


                # Returns a df with one column '{ticker}-rolling-beta'
                elif isinstance(self.hedge_instruments,str):
                    beta_data = beta_obj.get_rolling_beta()

                    self.rolling_beta_df = beta_data.copy()
                    self.rolling_beta_df_cols = ['rolling_beta']

                    self.latest_beta = float(beta_data.copy()['rolling_beta'].iloc[-1])

                    # Today's beta is yesterday's one to prevent look-ahead bias
                    beta_data['rolling_beta'] = beta_data['rolling_beta'].shift(1)

                    beta_data.rename(columns={'rolling_beta': 'rolling-beta'}, inplace=True)


                # Drop the first observation since it will be NaN after shifting by 1
                beta_data.dropna(ignore_index=True, inplace=True)


                beta_data = beta_data[beta_data["date"] >= pd.Timestamp(self.backtest_start_date)].copy()
                beta_data = beta_data[beta_data["date"] <= pd.Timestamp(self.backtest_end_date)].copy()


                

                # Update the actual start and end of backtest to what is available in the dataframe with all merged by date
                self.effective_backtest_start_date = beta_data['date'].iloc[0].strftime('%Y-%m-%d')
                self.effective_backtest_end_date = beta_data['date'].iloc[-1].strftime('%Y-%m-%d')


                # This dataframe gives the actual betas to use on each rebalance date
                beta_readings = beta_data.iloc[::self.rebalance_freq].copy()

                beta_readings = beta_readings.reset_index(drop=True)


            # Returns either a float or a dictionary mapping {ticker}: {beta}   
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
        default = self.WINDOW_DEFAULTS[self.freq]
        floor = self.WINDOW_FLOORS[self.freq]

        if window is None:
            return default

        if not isinstance(window, int) or window < floor:
            raise ValueError(
                f"window must be an integer >= {floor} observations for a "
                f"statistically reliable rolling regression at '{self.freq}' "
                f"frequency; got {window}."
            )

        return window


    def _date_resolver(self, backtest_start_date, backtest_end_date, backtest_period):
        # Compute start date
        
        today = date.today()

        self.data_start_date = '1800-01-01'

        if backtest_period is not None:
            if backtest_period in self.ALLOWED_PERIODS:
                self.backtest_start_date = (today - self.ALLOWED_PERIODS[backtest_period]).strftime("%Y-%m-%d")
                self.backtest_end_date = today.strftime("%Y-%m-%d")
            else:
                raise ValueError("backtest_period of the wrong format! choose from 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 20y, 30y.")
        else:
            if backtest_start_date is not None:
                self.backtest_start_date = (date.fromisoformat(backtest_start_date)).strftime("%Y-%m-%d")
            else:
                self.backtest_start_date = '1800-01-01'

            if backtest_end_date is not None:
                self.backtest_end_date = (date.fromisoformat(backtest_end_date) - timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                self.backtest_end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")


    def _rebalance_freq_resolver(self, rebalance_freq: int | None) -> int:

        floor = 1  # can't rebalance more often than every bar; no estimator-stability
                   # argument applies here since this only governs how often an
                   # already-fitted beta is resampled, not regression sample size

        if rebalance_freq is None:
            return max(self.rolling_window // 6, floor)

        if not isinstance(rebalance_freq, int) or rebalance_freq < floor:
            raise ValueError(
                f"rebalance_freq must be a positive integer (>= {floor}); "
                f"got {rebalance_freq}."
            )

        return rebalance_freq
        


    def _rolling_beta_plot(self, ax=None):

        if self.hedge_type == 'rolling':

            rolling_beta_df = self.rolling_beta_df
            rolling_beta_df_cols = self.rolling_beta_df_cols

        else:
            raise ValueError('cannot plot rolling beta for static hedge!')


        if ax is None:
            ax = plt.gca() # Fallback to the current active axes if none is provided


        for col in rolling_beta_df_cols:
            ax.plot(rolling_beta_df["date"], rolling_beta_df[col], label=col)


        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling beta")
        ax.legend()
        ax.grid(True, alpha=0.2)

        ax.set_title(
            f"Rolling beta with respect to each hedge instrument",
            pad=10
        )

        ax.text(0.5, 1.02, f"Rolling window: {self.rolling_window} observations", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="gray")

        ax.axhline(y=0, alpha = 0.3, linestyle='--', color='gray')
        
        return ax


    def _backtest_plot(self,ax=None):

        if ax is None:
            ax = plt.gca() # Fallback to the current active axes if none is provided


        # Plots the price series of the hedged and unhedged portfolio over time


        
        
        if not hasattr(self, 'hedged_portfolio_price_series'):
            self._port_val_series_hedged(portfolio_starting_value=100)

        hedged_price_series = self.hedged_portfolio_price_series

        unhedged_price_series = self.unhedged_portfolio_price_series  # guaranteed set as a
                                                                      # side effect of the call above



        
        # Plotting the hedged and unhedged indices
        df1 = unhedged_price_series
        df2 = hedged_price_series

        label1 = 'unhedged portfolio'
        label2 = 'hedged portfolio'




        ax.plot(df1["date"], df1['port-value'], label=label1)
        ax.plot(df2["date"], df2['port-value'], label=label2)


        ax.set_xlabel("Date")
        ax.set_ylabel("Index")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if self.hedge_type == 'rolling':
            ax.set_title(
                f"Hedged vs Unhedged Portfolio (index start = 100)",
                
                pad = 20
            )
        else:
            ax.set_title(
                "Hedged vs Unhedged Portfolio",
                pad = 20
            )

        if self.hedge_type == 'rolling':
            ax.text(0.5, 1.02, f"Rolling window: {self.rolling_window}; Rebalance: every {self.rebalance_freq} observations", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="gray")


        
        return ax
    

    def _plot(self):


        if self.hedge_type == 'rolling':

            self._compute_hedge_ratio()  # idempotent; ensures rolling_beta_df and
                                         # effective_backtest_start/end_date are resolved
                                         # before either subplot, regardless of call order between backtest/backtest_plot


            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16, 14))

            self._rolling_beta_plot(ax=ax1)
            self._backtest_plot(ax=ax2)

        elif self.hedge_type == 'static':
            fig, ax = plt.subplots(figsize= (10,4) )
            self._backtest_plot(ax=ax)


        fig.autofmt_xdate()
        plt.tight_layout()

        plt.show()
        plt.subplots_adjust(hspace=0.4) 

        if self.hedge_type == 'rolling':
            return fig, (ax1, ax2)
        elif self.hedge_type == 'static':
            return fig, ax



        




if __name__ == '__main__':
    portfolio = PortfolioHedge(
        target= {'nke':20,'ko':20, 'aapl':20, 'goog':20, 'nvda':20},
        hedge_instruments= ['spy','qqqm'],
        backtest_start_date= '2018-09-09',
        backtest_end_date= '2025-09-09',
        frequency='daily',
        return_type='simple',
        hedge_type='rolling'
    )

    portfolio.backtest()
    #portfolio.backtest_plot()

