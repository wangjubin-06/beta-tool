import requests
import pandas as pd
import os
from pathlib import Path



class FredApi:

     #----------------------------------------------------------
        #       FRED API
        #
        # DOCUMENTATION:
        # https://fred.stlouisfed.org/docs/api/fred/series_observations.html
        #
        #
        # the api here returns data with column 'date' in the form of a pd df datetime object, and 'value' in the form of floats that are the actual time series data
        #
        #----------------------------------------------------------
    

    def __init__(self, api_key, frequency):

        allowed_freq = {
            "d",
            "w",
            "bw",
            "m",
            "q",
            "sa",
            "a"
        }
        if not frequency in allowed_freq:
            raise ValueError("input valid data frequencies: d, w, bw, m, q, sa, a")
        
        self.api_key = api_key

        self.freq = frequency

        # frequencies allowed:
        # d = Daily
        # w = Weekly
        # bw = Biweekly
        # m = Monthly
        # q = Quarterly
        # sa = Semiannual
        # a = Annual


    def _get_full_history(self, ticker):
        """
        Return full historical data
        """
        
        df = self._fred_data(ticker = ticker)
        
        return df

    def _get_history(self, ticker, start=None, end=None):
        """
        Return data between start and end.
        """
        
        if end is None:
            df = self._fred_data(ticker = ticker, start_date=start)
        else:
            df = self._fred_data(ticker = ticker, start_date=start, end_date=end)
    
        return df

    def _fred_data(self, ticker, start_date=None, end_date=None):
    
        url = "https://api.stlouisfed.org/fred/series/observations"

        params = {
                "api_key": self.api_key,
                "file_type": "json",
                "series_id": f"{ticker.upper()}",
                "frequency": self.freq
            }
        
        if start_date is None and end_date is None:
            params = params  
        elif start_date is None and end_date is not None:
            params["observation_end"] = end_date
        elif end_date is None and start_date is not None:
            params["observation_start"] = start_date
        else:
            params["observation_end"] = end_date
            params["observation_start"] = start_date
        

        response = requests.get(url, params=params)

    
        response.raise_for_status()

        df = pd.DataFrame(response.json()['observations'])

    
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[['date', 'value']]
        df["date"] = pd.to_datetime(df["date"])


        return df



    def get_data(self, ticker, start_date, end_date = None):
        
        CACHE_DIR = Path("data/fred")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
        cache_file = CACHE_DIR / f"{ticker}_{self.freq}.parquet"
    
        # -------------------------
        # 1. Load existing cache
        # -------------------------
        if cache_file.exists():
            cached = pd.read_parquet(cache_file)
            cached["date"] = pd.to_datetime(cached["date"])
        else:
            cached = pd.DataFrame()

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
            df = self._get_history(ticker, start_date, end_date)
    
            df["date"] = pd.to_datetime(df["date"])
    
            df.to_parquet(
                cache_file,
                engine="pyarrow",
                index=False,
            )
    
            return df
    
        # -------------------------
        # 3. Figure out what's missing
        # -------------------------
            
            
        cached_start = cached["date"].min()
        cached_end = cached["date"].max()
    
        pieces = [cached]
    
        # Missing data BEFORE cache
        if requested_start < cached_start:
            fetch_end = cached_start - pd.Timedelta(days=1)
    
            old_start = requested_start.strftime("%Y-%m-%d")
            old_end = fetch_end.strftime("%Y-%m-%d")
    
            older = self._get_history(ticker, old_start, old_end)
            older["date"] = pd.to_datetime(older["date"])
    
            pieces.append(older)
    
        # Missing data AFTER cache
        if requested_end > cached_end:
            fetch_start = cached_end + pd.Timedelta(days=1)
    
            new_start = fetch_start.strftime("%Y-%m-%d")
            new_end = requested_end.strftime("%Y-%m-%d")
    
            newer = self._get_history(ticker, new_start, new_end)
    
            newer["date"] = pd.to_datetime(newer["date"])
    
            pieces.append(newer)
    
    
        # -------------------------
        # 4. Merge + save cache
        # -------------------------
        result = (
            pd.concat(pieces, ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
    
        result.to_parquet(
            cache_file,
            engine="pyarrow",
            index=False,
        )
    
        # Return only what caller requested
        return result[
            (result["date"] >= requested_start)
            & (result["date"] <= requested_end)
        ]

if __name__ == "__main__":
    api_key = os.getenv('FRED_API_KEY')
    ticker = "dgs3mo"
    fred_obj = FredApi(api_key, "m")

    data = fred_obj.get_data(ticker, "2026-01-01")
    print(data.head())