import requests
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta, date



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
                "series_id": ticker.upper(),
                "frequency": self.freq
            }
        
        if start_date is not None:
            params["observation_start"] = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        if end_date is not None:
            params["observation_end"] = pd.Timestamp(end_date).strftime("%Y-%m-%d")
            
        response = requests.get(url, params=params)

        try:
            response.raise_for_status()

        except requests.HTTPError as exc:

            raise RuntimeError(
                f"Fred HTTP error for {ticker}.\n"
                f"Status: {response.status_code}\n"
                f"URL: {response.url}\n"
                f"Response: {response.text[:500]}"
            ) from exc




        data = response.json()
        
        
        if "observations" not in data:
            raise RuntimeError(
                f"Unexpected FRED response:\n{data}"
            )

        observations = data["observations"]

        if not observations:
            return pd.DataFrame(columns=["date", "value"])
    

        df = pd.DataFrame(observations)
        
        required_columns = ['date','value']
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:

            raise RuntimeError(
                f"Fred returned unexpected columns for {ticker}.\n"
                f"Missing: {missing_columns}\n"
                f"Received: {df.columns.tolist()}\n"
            )
                    
                    
        df = df[["date", "value"]]

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        df["date"] = pd.to_datetime(df["date"])
        
        df.dropna(ignore_index=True, inplace=True)
        
        return df

    def get_data(self, ticker, start_date= None, end_date = None):
        """
        Return Fred historical data for the requested date range.
        
        Data is cached locally in Parquet files.

        If the requested range extends beyond the cached range,
        only the missing portion is downloaded.

        Args:
            ticker (str): ticker of series data
            start_date (str, optional): start date of requested data. Defaults to None.
            end_date (str, optional): end date of requested data. Defaults to None.

        Returns:
            pandas Dataframe: a df of the data in the requested date range
        """
        
        
        
        cache_file = self._get_cache_file(ticker)
        
        if start_date is None:
            requested_start = pd.Timestamp("1800-01-01")
            #requested_start_str = requested_start.strftime('%Y-%m-%d')
        else:
            requested_start = pd.Timestamp(start_date)
            #requested_start_str = requested_start.strftime('%Y-%m-%d')

        if end_date is None:
            requested_end = pd.Timestamp.today()
            #requested_end_str = requested_end.strftime('%Y-%m-%d')
        else:
            requested_end = pd.Timestamp(end_date)
            #requested_end_str = requested_end.strftime('%Y-%m-%d')
            
        
        if requested_start > requested_end:
            raise ValueError(
                f"start_date ({requested_start.date()}) cannot be after end_date ({requested_end.date()})"
            )
                    
                    
        # -------------------------
        # 1. Load existing cache
        # -------------------------
        
        cached = self._load_cache(cache_file)
        
        # -------------------------
        # 2. If cache is empty,
        #    fetch everything
        # -------------------------
                
        if cached.empty:
            df = self._get_history(
                ticker=ticker,
                start=requested_start,
                end=requested_end,
            )

            # Don't cache an empty response.
            #
            # This is important because an empty response can be
            # caused by requesting a future/weekend-only period.
            if df.empty:
                return df

            self._save_cache(df, cache_file)

            return self._filter_date_range(
                df,
                requested_start,
                requested_end,
            )
            
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
    
            older = self._get_history(ticker = ticker, start = old_start, end = old_end)
            
            # ensure that append only if older has actual values
            #
            # because requested dates may not be a trading day/may be beyond the date range in FRED,
            # so requested data may be empty
            #
            if not older.empty:
                older["date"] = pd.to_datetime(older["date"])
                pieces.append(older)
    
        # Missing data AFTER cache
        if requested_end > cached_end:
            
            # --------------------------------------------------
            # If requested_end is a weekend, don't bother making
            # an API request just for Saturday/Sunday.
            #
            # We still fetch up to the most recent weekday.
            # --------------------------------------------------
                        
            fetch_end = self._latest_weekday(requested_end)
            
            fetch_start = cached_end + pd.Timedelta(days=1)
            
            if fetch_start <= fetch_end:
                
                newer = self._get_history(
                    ticker = ticker,
                    start= fetch_start.strftime("%Y-%m-%d"),
                    end = fetch_end.strftime("%Y-%m-%d")
                )
                
                if not newer.empty:
                    # ensure that append only if newer has actual values
                    #
                    # because requested dates may be beyond the date range in FRED,
                    # so requested data may be empty
                                
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
        
        # Save updated cache
        if not result.empty:
            self._save_cache(result, cache_file)
    
        # Return only what caller requested
        return result[
            (result["date"] >= requested_start)
            & (result["date"] <= requested_end)
        ]

    def _get_cache_file(self, ticker):
        """ Return the cache path for a ticker

        Args:
            ticker (str)
        """
        cache_dir = Path("../../data/fred")
        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return cache_dir / (
            f"{ticker}_{self.freq}.parquet"
        )
    
    def _load_cache(self, cache_file):
        """
        Load an existing Parquet cache.
        Invalid/empty cache files are treated as no useful cache.
        
        """
        
        if not cache_file.exists():
            return pd.DataFrame()
        
        try:
            cached = pd.read_parquet(cache_file)
        
        except Exception as exc:
            
            raise RuntimeError(
                f"Unable to read Fred cache:\n"
                f"{cache_file}"
            ) from exc
        
        if cached.empty:
            return pd.DataFrame()
        
        if "date" not in cached.columns:
            raise RuntimeError(
                f"Fred cache is missing the 'date' column:\n"
                f"{cache_file}"
            )
        
        cached["date"] = (
            pd.to_datetime(cached["date"])
            .dt.normalize()
        )

        return (
            cached
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

    def _save_cache(self, df, cache_file):
        """
        Save DataFrame to Parquet cache.
        """

        if df.empty:
            return

        df = (
            df
            .copy()
            .sort_values("date")
            .drop_duplicates(subset=["date"])
            .reset_index(drop=True)
        )

        df.to_parquet(
            cache_file,
            engine="pyarrow",
            index=False,
        )



    @staticmethod
    def _filter_date_range(
        df,
        start_date,
        end_date,
    ):
        """
        Return only rows inside requested date range.
        """

        if df.empty:
            return df.copy()

        return (
            df[
                (df["date"] >= start_date)
                & (df["date"] <= end_date)
            ]
            .reset_index(drop=True)
        )

    @staticmethod
    def _latest_weekday(timestamp):
        """
        Return timestamp if it is a weekday.

        If Saturday/Sunday, roll backwards to Friday.

        This prevents unnecessary API requests on weekends.
        """

        timestamp = pd.Timestamp(timestamp).normalize()

        while timestamp.weekday() >= 5:
            timestamp -= pd.Timedelta(days=1)

        return timestamp



          
if __name__ == "__main__":
    api_key = os.getenv('FRED_API_KEY')
    ticker = "dgs3mo"
    fred_obj = FredApi(api_key, "m")
    data = fred_obj.get_data(ticker=ticker, start_date="2026-01-01")
    print(data)