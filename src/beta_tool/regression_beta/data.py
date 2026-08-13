"""
    this module gathers past asset price data using the London Strategic Edge API. requires the user to use their own API key stored in their system environments

"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from lse import LSE

class AssetData:

    def __init__(
        self,
        ticker: str,
        period: str ="1y",
        interval: str = "daily",
        start_date: date | None = None,
        end_date: date | None = None
        ):

        self.ticker = ticker.upper()
        self.start_date, self.end_date = AssetData._resolve_dates(period, start_date, end_date)
        self.interval = AssetData._interval_resolver(interval)

    @staticmethod
    def _resolve_dates(
        period: str,
        start_date: date | None,
        end_date: date | None,
        ) -> tuple[date,date]:

        """

        period logic:
        if end date is provided, end of observation is the end date provided
        if start date is provided, start of observation is the start date provided

        if end date is not provided, end of observation is the date today
        if start date is not provided, start date is end of observation minus the period provided
        if period is NOT provided, it defaults to 1 year (1y)
        
        start date takes precendence over period provided.


        """

        if end_date is not None:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            end = end_date
        else:
            end = date.today()

        if start_date is not None:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            start = start_date - timedelta(days = 1)
        else:
            periods = {
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

            if period not in periods:
                raise ValueError(f"Unsupported period: {period}")

            start = end - periods[period] - timedelta(days=1)
            #start date is 1 day before the actual start date so pd df can calculate pct change for first day

        if start > end:
            raise ValueError("start_date cannot be after end_date")

        return start, end

    # @staticmethod
    # def _stooq_ticker(ticker, exchange):
    #     ticker = ticker.lower()
    #     if type(ticker) != str or type(exchange) != str:
    #         raise ValueError("input correct string format for stock ticker and exchange.")
        
    #     if "." in ticker:
    #         raise ValueError("stock ticker should not have '.'. use '-' instead if reffering to share class. for example: BRK-B will refer to class B shares of Berkshire Hathaway")

    #     exchanges = {"us","l","to","ax","hk","de","pa","sz","ss","si"}

    #     if not exchange.lower() in exchanges:
    #         raise ValueError("stock exchange suffix does not exist")

    #     exchange = exchange.lower()

    #     return f"{ticker}.{exchange}"

    @staticmethod
    def _interval_resolver(interval):
        interval = interval.lower()
        interval_dict = {
            "daily": "1d",
            "weekly": "1w",
            "monthly": "1mo",
        }

        if interval in interval_dict:
            return interval_dict[interval]
        elif interval in interval_dict.values():
            return interval
        else:
            raise ValueError("interval is not of the correct format. options available: daily, weekly, monthly")


    def get_prices(self):
        client = LSE(api_key=os.environ.get('LSE_API_KEY')) #User has to sign up for an account at https://londonstrategicedge.com/ and save their own api key in their system's environment variables under the name 'LSE_API_KEY'
        candles = client.candles(self.ticker, self.interval, self.start_date, self.end_date)
        df = pd.DataFrame(candles)


        #cleaning up the timestamp strings provided by LSE api to become year-month-day format and converting them to pd datetime objects
        df["timestamp"] = pd.to_datetime(df["timestamp"].str.split('T').str[0], format='%Y-%m-%d', errors = "coerce")

        # Basic daily equity sanity checks
        df = df[
            (df["timestamp"].dt.dayofweek < 5) &
            (df["volume"] > 0) &
            (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        ].copy()

        df = df.sort_values("timestamp").reset_index(drop=True)



        #calculating adjusted close instead of raw candle close

        # --------------------------------------------------
        # 2. Get dividends
        # --------------------------------------------------

        dividends = client.dividends(
            symbol=self.ticker,
            order="asc",
            limit=5000
        )

        div_df = pd.DataFrame(dividends)
        if not div_df.empty:
            div_df["effective_date"] = pd.to_datetime(
                div_df["effective_date"]
            )
        # --------------------------------------------------
        # 3. Start with adjustment factor = 1
        # --------------------------------------------------
        df["adjustment_factor"] = 1.0

        # --------------------------------------------------
        # 4. Apply dividend adjustments backwards - a dividend payout artificially lowers the price of a stock; no real return loss, hence we need to adjust the raw closing price to reflect this. For example, price is $100 the day before dividend payout, and dividend is $1. the ex-dividend price is $99, so all prices before the dividend date and after the previous dividend date have to be multiplied by a factor of 99/100.
        # --------------------------------------------------
        if not div_df.empty:
            for _, dividend in div_df.iterrows():
                ex_date = dividend["effective_date"]
                amount = float(dividend["dividend_amount"])

                # Find the last trading day BEFORE ex-date
                previous = df.loc[
                    df["timestamp"] < ex_date,
                    "close"
                ]

                if previous.empty:
                    continue

                previous_close = previous.iloc[-1]

                if previous_close <= 0:
                    continue

                factor = (
                    previous_close - amount
                ) / previous_close

                # Apply the dividend factor to all
                # prices before the ex-date

                # Dividend adjustment applies only to observations
                # strictly BEFORE the ex-dividend date.
                df.loc[
                    df["timestamp"] < ex_date,
                    "adjustment_factor"
                ] *= factor

        # --------------------------------------------------
        # 5. LSE's Data is already adjusted for splits
        # --------------------------------------------------

        # --------------------------------------------------
        # 6. Calculate adjusted close
        # --------------------------------------------------
        df["adjusted_close"] = (
            df["close"] *
            df["adjustment_factor"]
        )

        ret_df = df.copy()
        ret_df["timestamp"] = pd.to_datetime(ret_df["timestamp"], errors="coerce")

        ret_df = (
                ret_df.dropna(subset=["timestamp", "adjusted_close"])
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="last")
                .reset_index(drop=True)
            )



        return ret_df


if __name__ == "__main__":
    client = LSE(api_key=os.environ.get('LSE_API_KEY'))

    apple = AssetData(ticker= "aapl", interval="daily", period="1y") 
    data = apple.get_prices()

    print(data.head())