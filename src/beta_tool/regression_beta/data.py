"""
    this module gathers past asset price data using Tiingo API. requires the user to use their own API key stored in their system environments

"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser
from data_collection.tiingo_api import TiingoApi

class AssetData:

    ALLOWED_FREQUENCIES = {
        'daily',
        'weekly',
        'monthly',
        'annually'
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
        ticker: str,
        period: str ="1y",
        frequency: str = "daily",
        start_date: str | None = None,
        end_date: str | None = None
        ):

        if frequency not in self.ALLOWED_FREQUENCIES:
            raise ValueError("only daily, weekly, monthly, annually is allowed for data interval!")

        if period not in self.ALLOWED_PERIODS:
            raise ValueError("choose from: 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 20y, 30y for 'period'!")

        self.freq = frequency

        self.ticker = ticker
        self.start_date, self.end_date = self._resolve_dates(period, start_date, end_date)


    def _resolve_dates(
        self,
        period: str,
        start_date: str | None,
        end_date: str | None,
        ):

        """

        period logic:
        if end date is provided, end of observation is the end date provided
        if start date is provided, start of observation is the start date provided

        start_date (provided) + period = end_date (not provided)
        end_date (provided) - period = start_date (not provided)

        if start_date & end_date are provided; period is ignored.

        if end date is not provided, end of observation is the date today
        if start date is not provided, start date is end of observation minus the period provided
        if period is NOT provided, it defaults to 1 year (1y)
        
        start date takes precendence over period provided.

        """

        if end_date is not None:
            try:
                # Automatically parses almost any date format into a datetime object
                parsed_date = parser.parse(end_date)
                # Formats the datetime object into strictly 'yyyy-mm-dd'
                end = parsed_date.date()
            except (ValueError, TypeError):
                return None  # Handles invalid date strings gracefully
        else:
            end = date.today()

        if start_date is not None:
            try:
                # Automatically parses almost any date format into a datetime object
                parsed_date = parser.parse(start_date)
                # Formats the datetime object into strictly 'yyyy-mm-dd'
                start = parsed_date.date()
            except (ValueError, TypeError):
                return None  # Handles invalid date strings gracefully
            start = start - timedelta(days = 1)
        else:
            start = end - self.ALLOWED_PERIODS[period] - timedelta(days=1)
            #start date is 1 day before the actual start date so pd df can calculate pct change for first day

        if start > end:
            raise ValueError("start_date cannot be after end_date")

        return str(start), str(end)


    def get_prices(self):

        api_key = os.getenv("TIINGO_API_KEY")
        
        tiingo = TiingoApi(
            api_key=api_key,
            frequency=self.freq,
            simplified=True,
        )
    
        data = tiingo.get_data(
            ticker = self.ticker,
            start_date = self.start_date,
            end_date = self.end_date
        )

        return data


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
    pass