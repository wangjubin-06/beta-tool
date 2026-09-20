import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from .errors import *
from datetime import timedelta, date, datetime
import pandas as pd
import requests
from io import StringIO
from pathlib import Path
import io
import zipfile

logger = logging.getLogger(__name__)


# ==============================================================
# Cache location
# ==============================================================

def _cache_root() -> Path:
    """
    Directory holding the Tiingo Parquet cache.

    Set the BETA_TOOL_DATA_DIR environment variable to relocate it;
    the cache lives in a 'tiingo' subfolder of that directory.

    Defaults to ./data relative to the current working directory
    (the project root when launching with `uv run streamlit run ...`).

    Resolved at call time (not import time) so tests can point it at a
    temporary directory.
    """

    base = os.getenv("BETA_TOOL_DATA_DIR")
    root = Path(base) if base else Path.cwd() / "data"

    return root / "tiingo"


# ==============================================================
# Per-session API key override (for hosted apps)
# ==============================================================

_key_override: ContextVar[str | None] = ContextVar(
    "tiingo_key_override",
    default=None,
)


@contextmanager
def tiingo_key_override(key):
    """
    Use `key` for every TiingoApi created inside this block.

    The override is stored in a ContextVar, so it only applies to the
    current thread / Streamlit session and never leaks to other visitors
    (unlike setting os.environ, which is shared by the whole process).

    While active, it takes precedence over any key passed to TiingoApi.
    An empty / None key means "no override".

    Example (in a Streamlit page):

        with tiingo_key_override(user_key):
            result = Beta(...)
    """

    token = _key_override.set(key or None)

    try:
        yield
    finally:
        _key_override.reset(token)


class TiingoApi:
    """
    Interface to the Tiingo End-of-Day API with local Parquet caching.

    Supported frequencies:
        - daily
        - weekly
        - monthly
        - annually

    Parameters
    ----------
    api_key : str
        Tiingo API key. Ignored if a tiingo_key_override(...) block is
        active, in which case the override key is used instead.

    frequency : str
        One of:
            "daily"
            "weekly"
            "monthly"
            "annually"

    simplified : bool
        If True, only return:
            date
            adjClose

        If False, return the full Tiingo price dataset.
    """

    FULL_COLUMNS = [
        "date",
        "close",
        "high",
        "low",
        "open",
        "volume",
        "adjClose",
        "adjHigh",
        "adjLow",
        "adjOpen",
        "adjVolume",
        "divCash",
        "splitFactor",
    ]

    SIMPLE_COLUMNS = [
        "date",
        "adjClose",
    ]

    ALLOWED_FREQUENCIES = {
        "daily",
        "weekly",
        "monthly",
        "annually",
    }

    # How often we are willing to check Tiingo for new data
    REFRESH_INTERVALS = {
        "daily": timedelta(hours=24),
        "weekly": timedelta(days=4),
        "monthly": timedelta(days=15),
        "annually": timedelta(days=120),
    }
    
    def __init__(self, api_key, frequency, simplified=False):

        if frequency not in self.ALLOWED_FREQUENCIES:
            raise ValueError(
                "input valid data frequencies: "
                "daily, weekly, monthly, annually"
            )
            
        # A per-session override (see tiingo_key_override) wins over
        # whatever key the caller passed in.
        api_key = _key_override.get() or api_key
        
        api_key = require_api_key(api_key, "TIINGO_API_KEY", "Tiingo")

        if not api_key:
            raise ValueError(
                "Tiingo API key is missing. "
                "Set the TIINGO_API_KEY environment variable."
            )

        self.api_key = api_key
        self.freq = frequency
        self.simplified = simplified
        
        self.refresh_interval = self.REFRESH_INTERVALS[frequency]

    # ==========================================================
    # Public API
    # ==========================================================

    def get_data(self, ticker, start_date=None, end_date=None):
        """
        Return Tiingo historical data for the requested date range.

        Data is cached locally in Parquet files.

        The cache will only be updated when the cache becomes stale
        dictated by self.REFRESH_INTERVALS, so the cache data date range
        may not be the latest.
        
        This explicit design is chosen to balance
        between conserving API call limits and having enough data points
        for regression.

        Writing the cache is best-effort: if the cache folder is missing,
        read-only or otherwise unwritable (e.g. on a hosted app), a warning
        is logged and the freshly downloaded data is used directly.
        
        """

        ticker = ticker.lower()

        cache_file = self._get_cache_file(ticker)

        # ------------------------------------------------------
        # Requested date range
        # ------------------------------------------------------

        requested_start = (
            pd.Timestamp("1960-01-01") if start_date is None else pd.Timestamp(start_date)
        )

        requested_end = (
            pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date)
        )

        if requested_start > requested_end:
            raise ValueError(
                f"start_date ({requested_start.date()}) cannot be after end_date ({requested_end.date()})"
            )


        # ------------------------------------------------------
        # Cache missing or stale -> download full history
        # ------------------------------------------------------

        if self._cache_is_stale(cache_file):

            df = self._get_full_history(ticker)

            if df.empty:
                return self._empty_dataframe()

            # Best-effort: a failed write must not fail the request.
            self._save_cache(df, cache_file)

            # Use the frame we just downloaded instead of re-reading
            # it from disk.
            data = df

        # ------------------------------------------------------
        # Cache is fresh -> use local data
        # ------------------------------------------------------

        else:
            data = self._load_cache(cache_file)
        
        return self._filter_date_range(
            data,
            requested_start,
            requested_end,
        )

    # ==========================================================
    # Historical data helpers
    # ==========================================================

    def _get_full_history(self, ticker):
        """
        Return full available historical data.
        """

        return self._get_history(
            ticker=ticker,
            start=None,
            end=None,
        )

    def _get_history(self, ticker, start=None, end=None):
        """
        Return historical data between start and end.
        """

        return self._tiingo_data(
            ticker=ticker,
            start_date=start,
            end_date=end,
        )

    # ==========================================================
    # Tiingo API request
    # ==========================================================

    def _tiingo_data(self,ticker,start_date=None,end_date=None):
        """
        Make a direct request to Tiingo.

        Important:
        Tiingo can return HTTP 200 even when the response body
        contains an API error such as an exceeded request quota.

        Therefore we explicitly inspect the response body.
        """

        ticker = ticker.lower()

        url = (
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

        # ------------------------------------------------------
        # Dates
        # ------------------------------------------------------

        if start_date is None:
            start_date = "1960-01-01"
        else:
            start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = date.today().isoformat()
        else:
            end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")

        # ------------------------------------------------------
        # Request parameters
        # ------------------------------------------------------

        params = {
            "startDate": start_date,
            "endDate": end_date,
            "resampleFreq": self.freq,
            "sort": "date",
            "format": "csv",
        }

        if self.simplified:
            # IMPORTANT:
            # Send this as a string rather than a Python list.
            params["columns"] = "date,adjClose"

        # ------------------------------------------------------
        # Make request
        # ------------------------------------------------------

        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise DataError(f"Could not reach Tiingo for {ticker}: {exc}") from None
        
        
        check_response(response, provider="Tiingo", var="TIINGO_API_KEY", what=ticker)
        check_body(response, provider="Tiingo", var="TIINGO_API_KEY", what=ticker)
        
        
        text = response.text.strip()

        # # ------------------------------------------------------
        # # HTTP-level errors
        # # ------------------------------------------------------

        # try:
        #     response.raise_for_status()

        # except requests.HTTPError as exc:

        #     raise RuntimeError(
        #         f"Tiingo HTTP error for {ticker}.\n"
        #         f"Status: {response.status_code}\n"
        #         f"URL: {response.url}\n"
        #         f"Response: {response.text[:500]}"
        #     ) from exc

        # # ------------------------------------------------------
        # # Body-level errors
        # #
        # # Tiingo can return HTTP 200 while putting the error
        # # message in the response body.
        # # ------------------------------------------------------

        # text = response.text.strip()

        # if text.lower().startswith("error:"):

        #     raise RuntimeError(
        #         f"Tiingo API error for {ticker}.\n"
        #         f"Date range: {start_date} -> {end_date}\n"
        #         f"Response: {text}"
        #     )

        # ------------------------------------------------------
        # Empty response
        # ------------------------------------------------------

        if not text:
            return self._empty_dataframe()

        # ------------------------------------------------------
        # Parse CSV
        # ------------------------------------------------------

        try:
            df = pd.read_csv(StringIO(text))

        except Exception as exc:

            raise RuntimeError(
                f"Unable to parse Tiingo response for {ticker}.\n"
                f"URL: {response.url}\n"
                f"Response: {text[:500]}"
            ) from exc

        # ------------------------------------------------------
        # Legitimately empty dataset
        # ------------------------------------------------------

        if df.empty:
            return self._empty_dataframe()

        # ------------------------------------------------------
        # Validate columns
        # ------------------------------------------------------

        required_columns = (
            self.SIMPLE_COLUMNS if self.simplified else self.FULL_COLUMNS
        )

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:

            raise RuntimeError(
                f"Tiingo returned unexpected columns for {ticker}.\n"
                f"Missing: {missing_columns}\n"
                f"Received: {df.columns.tolist()}\n"
                f"Response: {text[:500]}"
            )

        # ------------------------------------------------------
        # Keep requested columns
        # ------------------------------------------------------

        if self.simplified:
            df = df[
                [
                    "date",
                    "adjClose",
                ]
            ].copy()
        else:
            df = df[self.FULL_COLUMNS].copy()

        # ------------------------------------------------------
        # Clean data
        # ------------------------------------------------------

        df["date"] = (
            pd.to_datetime(df["date"])
            .dt.normalize()
        )

        df["adjClose"] = pd.to_numeric(
            df["adjClose"],
            errors="coerce",
        )

        # ------------------------------------------------------
        # Remove invalid rows
        # ------------------------------------------------------

        df = (
            df
            .dropna()
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        return df

    # ==========================================================
    # Cache
    # ==========================================================

    def _get_cache_file(self, ticker):
        """
        Return the cache path for a ticker.

        The cache folder is created if possible. If it cannot be created
        (read-only filesystem etc.) a warning is logged and the path is
        still returned: the cache then simply never exists, so data is
        downloaded each time instead of the request failing.
        """

        cache_dir = _cache_root()

        try:
            cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as exc:
            logger.warning(
                "Could not create Tiingo cache folder %s: %s",
                cache_dir,
                exc,
            )

        label = (
            "simple" if self.simplified else "detailed"
        )

        return cache_dir / (
            f"{ticker}_{self.freq}_{label}.parquet"
        )

    def _load_cache(self, cache_file):
        """
        Load an existing Parquet cache.

        Invalid/empty cache files are treated as no useful cache.
        """

        if not cache_file.exists():
            return self._empty_dataframe()

        try:
            cached = pd.read_parquet(cache_file)

        except Exception as exc:

            raise RuntimeError(
                f"Unable to read Tiingo cache:\n"
                f"{cache_file}"
            ) from exc

        if cached.empty:
            return self._empty_dataframe()

        if "date" not in cached.columns:
            raise RuntimeError(
                f"Tiingo cache is missing the 'date' column:\n"
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

        The write is atomic: data goes to a uniquely named temporary file
        in the same folder and is then moved into place with os.replace,
        so a concurrent reader never sees a half-written Parquet file.

        The write is also best-effort: any failure is logged as a warning
        and swallowed, because the cache is an optimisation, not a
        requirement for producing a result.
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

        tmp_file = cache_file.with_name(
            f"{cache_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )

        try:
            df.to_parquet(
                tmp_file,
                engine="pyarrow",
                index=False,
            )

            os.replace(tmp_file, cache_file)

        except Exception as exc:
            logger.warning(
                "Could not write Tiingo cache %s: %s",
                cache_file,
                exc,
            )

            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass

    # ==========================================================
    # Utility
    # ==========================================================

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

    def _empty_dataframe(self):
        """
        Return an empty DataFrame with the correct schema.
        """

        columns = (self.SIMPLE_COLUMNS if self.simplified else self.FULL_COLUMNS)

        return pd.DataFrame(columns=columns)

    def _cache_is_stale(self, cache_file):
        if not cache_file.exists():
            return True

        cache_age = datetime.now() - datetime.fromtimestamp(
            cache_file.stat().st_mtime
        )

        return cache_age >= self.REFRESH_INTERVALS[self.freq]

# ==============================================================
# Example
# ==============================================================

if __name__ == "__main__":

    api_key = os.getenv("TIINGO_API_KEY")

    tiingo = TiingoApi(
        api_key=api_key,
        frequency="daily",
        simplified=True,
    )

    data = tiingo.get_data(
        ticker="spy",
        start_date="2020-01-01",
    )

    print(data.head())
    print()
    print(data.tail())
    print()
    print("Rows:", len(data))
    print("Start:", data["date"].min())
    print("End:", data["date"].max())


# def get_tickers():
#     url = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
#     response = requests.get(url)
#     response.raise_for_status()

#     # Extract the CSV file from the ZIP archive
#     with zipfile.ZipFile(io.BytesIO(response.content)) as z:
#         # Find the CSV file inside the zip
#         csv_filename = [f for f in z.namelist() if f.endswith(".csv")][0]

#         with z.open(csv_filename) as f:
#             df = pd.read_csv(f)

#     return df