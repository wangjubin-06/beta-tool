"""
    this module gathers past asset price data using the London Strategic Edge API. requires the user to use their own API key stored in their system environments

"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
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

        end = end_date or date.today()

        if start_date is not None:
            start = start_date
        else:
            periods = {
                "1m": relativedelta(months=1),
                "3m": relativedelta(months=3),
                "6m": relativedelta(months=6),
                "1y": relativedelta(years=1),
                "2y": relativedelta(years=2),
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

    @staticmethod
    def _stooq_ticker(ticker, exchange):
        ticker = ticker.lower()
        if type(ticker) != str or type(exchange) != str:
            raise ValueError("input correct string format for stock ticker and exchange.")
        
        if "." in ticker:
            raise ValueError("stock ticker should not have '.'. use '-' instead if reffering to share class. for example: BRK-B will refer to class B shares of Berkshire Hathaway")

        exchanges = {"us","l","to","ax","hk","de","pa","sz","ss","si"}

        if not exchange.lower() in exchanges:
            raise ValueError("stock exchange suffix does not exist")

        exchange = exchange.lower()

        return f"{ticker}.{exchange}"

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
        df['timestamp'] = df['timestamp'].str[:10]
        df = df[["timestamp","symbol","close"]]
        return df


if __name__ == "__main__":
    apple = AssetData("aapl","2y","daily")
    data = apple.get_prices()