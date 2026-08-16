import requests
import pandas as pd
import io
import os
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta, date




class TiingoApi():

    #----------------------------------------------------------
    #       Tiingo API
    #
    # 80,000+ tickers (US Equities, ETFs, Mutual Funds, and Chinese A-Shares)
    #
    # adjusted close price that considers dividends and splits
    #
    # DOCUMENTATION:
    # https://www.tiingo.com/documentation/end-of-day
    #
    #----------------------------------------------------------

    # Meta Data
    #https://api.tiingo.com/tiingo/daily/<ticker>

    # Latest Price
    # https://api.tiingo.com/tiingo/daily/<ticker>/prices

    # Historical Prices
    #https://api.tiingo.com/tiingo/daily/<ticker>/prices?startDate=2012-1-1&endDate=2016-1-1 
    
    def __init__(self, api_key, frequency, simplified = False):
        allowed_freq = {
            "daily",
            "weekly",
            "monthly",
            "annually"
        }

        if not frequency in allowed_freq:
            raise ValueError("input valid data frequencies: daily, weekly, monthly, anually")

        self.api_key = api_key
        self.freq = frequency
        self.simplified = simplified

    def _get_full_history(self, ticker):
        """
        Return full historical data
        """

        df = self._tiingo_data(ticker=ticker).copy()

        return df
        

    def _get_history(self, ticker, start=None, end=None):
        """
        Return data between start and end.
        """

        if end is None:
            df = self._tiingo_data(ticker = ticker, start_date=start)
        else:
            df = self._tiingo_data(ticker = ticker, start_date=start, end_date=end)
            
        return df
    

    def _tiingo_data(self, ticker, start_date=None, end_date=None):

        asset_ticker = ticker.lower()
        url = f"https://api.tiingo.com/tiingo/daily/{asset_ticker}/prices"


        # interval_set = {"daily","weekly","monthly","annually"}
        # #the data will expand start and end to fit your entire interval windows; for example if you set start date on a wednesday, but interval is weekly, the start date will be rolled back to monday. end date will roll forward likewise
        # interval = "daily"



        headers = {
            'Content-Type': 'application/json',
            'Authorization' : f'Token {self.api_key}'
            }

        if start_date is not None and end_date is None:
            today = date.today().isoformat()
            params = {
                "startDate": start_date,
                "endDate": today,
                "resampleFreq": self.freq,
                "sort": "date",
                "format": "csv"
            }
        elif start_date is None and end_date is not None:
            params = {
                "startDate": "1800-01-01",
                "endDate": end_date,
                "resampleFreq": self.freq,
                "sort": "date",
                "format": "csv"
            }
        elif start_date is not None and end_date is not None:
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "resampleFreq": self.freq,
                "sort": "date",
                "format": "csv"
            }
        else:
            today = date.today().isoformat()
            params = {
                "startDate": "1800-01-01",
                "endDate": today,
                "resampleFreq": self.freq,
                "sort": "date",
                "format": "csv"
            }



        if self.simplified == True:
            params["columns"] = ["date","adjClose"]

        try:
            requestResponse = requests.get(url, headers=headers, params=params)
        
            requestResponse.raise_for_status()

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # e.g., 404 Client Error
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as err:
            print(f"An unexpected error occurred: {err}")

        df = pd.read_csv(io.BytesIO(requestResponse.content), encoding="utf-8")

        df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
        df['date'] = pd.to_datetime(df['date'])
        

        return df

    def get_data(self, ticker, start_date = None, end_date = None):

        # Tiingo Dates are timezone UNAWARE

        CACHE_DIR = Path("../../data/tiingo")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if self.simplified:
            label = "simple"
        else:
            label = "detailed"

        cache_file = CACHE_DIR / f"{ticker}_{self.freq}_{label}.parquet"
        CACHE_MAX_AGE = timedelta(hours=24)

        # -------------------------
        # 1. Load existing cache
        # -------------------------
        if cache_file.exists():
            modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)

            if datetime.now() - modified_time < CACHE_MAX_AGE:
                # Cache is still fresh
                cached = pd.read_parquet(cache_file)
                cached["date"] = pd.to_datetime(cached["date"])
            else:
                # Cache has expired
                cached = pd.DataFrame()
        else:
            cached = pd.DataFrame()

        if start_date is None:
            requested_start = None
        else:
            requested_start = pd.Timestamp(start_date)

        # If no end date supplied, use today
        if end_date is None:
            requested_end = pd.Timestamp.today().normalize()
        else:
            requested_end = pd.Timestamp(end_date)

        # -------------------------
        # 2. If cache is empty,
        #    fetch everything
        # -------------------------
        if cached.empty:
            df = self._get_full_history(ticker)

            df.to_parquet(
                cache_file,
                engine="pyarrow",
                index=False,
            )

            if requested_start is not None:
                return df[(df['date'] >= requested_start) & (df['date'] <= requested_end)]
            else:
                return df[(df['date'] <= requested_end)]

        if requested_start is not None:
            return cached[(cached['date'] >= requested_start) & (cached['date'] <= requested_end)]
        else:
            return cached[(cached['date'] <= requested_end)]

        # # -------------------------
        # # 3. Figure out what's missing
        # # -------------------------
        
        
        # cached_start = cached["date"].min()
        # cached_end = cached["date"].max()

        # pieces = [cached]

        # # Missing data BEFORE cache
        # if requested_start < cached_start:
        #     fetch_end = cached_start - pd.Timedelta(days=1)

        #     old_start = requested_start.strftime("%Y-%m-%d")
        #     old_end = fetch_end.strftime("%Y-%m-%d")

        #     older = self._get_history(ticker, old_start, old_end)
        #     older["date"] = pd.to_datetime(older["date"])

        #     pieces.append(older)

        # # Missing data AFTER cache
        # if requested_end > cached_end:
        #     fetch_start = cached_end + pd.Timedelta(days=1)

        #     new_start = fetch_start.strftime("%Y-%m-%d")
        #     new_end = requested_end.strftime("%Y-%m-%d")

        #     newer = self._get_history(ticker, new_start, new_end)

        #     newer["date"] = pd.to_datetime(newer["date"])

        #     pieces.append(newer)


        # # -------------------------
        # # 4. Merge + save cache
        # # -------------------------
        # result = (
        #     pd.concat(pieces, ignore_index=True)
        #     .drop_duplicates(subset=["date"])
        #     .sort_values("date")
        #     .reset_index(drop=True)
        # )

        # result.to_parquet(
        #     cache_file,
        #     engine="pyarrow",
        #     index=False,
        # )

        # Return only what caller requested
        # return result[
        #     (result["date"] >= requested_start)
        #     & (result["date"] <= requested_end)
        # ]


if __name__ == "__main__":
    api_key = os.getenv('TIINGO_API_KEY')
    ticker1 = "aapl"
    ticker2 = "aapl"
    t_obj_1 = TiingoApi(api_key, "monthly", True)
    t_obj_2 = TiingoApi(api_key, "daily", False)


    # data1 = t_obj_1.get_data(ticker1, "2026-01-01")
    # data2 = t_obj_2.get_data(ticker2, "2025-12-21", "2026-01-03")
    # print(data1.tail())
    # #print("\n"*5)
    # #print(data2.head())

    # print(data1.head())
    # print(data2.head())

    # data1[f'{ticker1}_simple_returns'] = (data1['adjClose'] / data1['adjClose'].shift(1)) - 1
    # data1[f'{ticker1}_log_returns'] = np.log(data1['adjClose'] / data1['adjClose'].shift(1))
    # df = data1[['date', f'{ticker1}_log_returns', f'{ticker1}_simple_returns']]
    # print(df.head())



