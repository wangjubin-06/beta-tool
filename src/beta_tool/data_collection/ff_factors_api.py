import logging
import os
import threading
import requests
import pandas as pd
from pathlib import Path
import io
import zipfile
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Seconds to wait for the Ken French download before giving up.
REQUEST_TIMEOUT = 60


# ==============================================================
# Cache location
# ==============================================================

def _cache_root() -> Path:
    """
    Directory holding the French factor Parquet cache.

    Uses the same BETA_TOOL_DATA_DIR environment variable as the other
    data modules, with a 'french' subfolder. Defaults to ./data relative
    to the current working directory (the project root when launching with
    `uv run streamlit run ...`).

    Resolved at call time so tests can point it at a temporary directory.
    """

    base = os.getenv("BETA_TOOL_DATA_DIR")
    root = Path(base) if base else Path.cwd() / "data"

    return root / "french"


class FrenchApi:

    #----------------------------------------------------------
    #       Fama-French factor portfolio API
    #
    #   DOCUMENTATION:
    #   https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    #
    #
    #   the static download links on the website returns 
    #   csv data 
    # 
    #   option to download daily factor returns or monthly
    #   monthly csv includes annual data
    #
    #----------------------------------------------------------

    
    # How often we are willing to check the French data library for new data
    REFRESH_INTERVALS = {
        "daily": timedelta(days=15),
        "monthly": timedelta(days=15),
        "annually": timedelta(days=180),
    }

    FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


    def __init__(self, region:str, frequency:str):

        regions = {
            'US',
            'Developed',
            'Developed ex-US',
            'Europe',
            'Japan',
            'APAC ex-Japan',
            'North America',
            'Emerging'
        }

        frequencies = {
            'daily',
            'monthly',
            'annually'
        }

        if region not in regions:
            raise ValueError("indicate asset region: 'US','Developed', 'Developed ex-US', 'Europe','Japan','APAC ex-Japan','North America','Emerging'")

        if frequency not in frequencies:
            raise ValueError("indicate frequency: 'daily', 'monthly', 'annually'")

        links_dic = {
            "daily": {
                'US': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
                'Developed': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_5_Factors_Daily_CSV.zip",
                'Developed ex-US': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_ex_US_5_Factors_Daily_CSV.zip",
                'Europe': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_Daily_CSV.zip",
                'Japan': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_5_Factors_Daily_CSV.zip",
                'APAC ex-Japan': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_5_Factors_Daily_CSV.zip",
                'North America': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/North_America_5_Factors_Daily_CSV.zip",
            },
            "monthly": {
                'US': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
                'Developed': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_5_Factors_CSV.zip",
                'Developed ex-US': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_ex_US_5_Factors_CSV.zip",
                'Europe': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_CSV.zip",
                'Japan': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_5_Factors_CSV.zip",
                'APAC ex-Japan': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_5_Factors_CSV.zip",
                'North America': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/North_America_5_Factors_CSV.zip",
                'Emerging': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Emerging_5_Factors_CSV.zip"
            },
            "annually": {
                'US': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
                'Developed': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_5_Factors_CSV.zip",
                'Developed ex-US': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_ex_US_5_Factors_CSV.zip",
                'Europe': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_CSV.zip",
                'Japan': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_5_Factors_CSV.zip",
                'APAC ex-Japan': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_5_Factors_CSV.zip",
                'North America': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/North_America_5_Factors_CSV.zip",
                'Emerging': "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Emerging_5_Factors_CSV.zip"
            }
        }

        self.freq = frequency
        self.region = region
        self.link = links_dic[frequency][region]

    def _get_full_history(self):
        """
        Return full historical data
        """

        if self.freq == 'daily':
            response = requests.get(self.link, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]

                if not csv_files:
                    raise ValueError("No CSV found in ZIP")

                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(
                        f,
                        skiprows=3,
                        skipfooter=1,
                        engine="python"
                    )

            df = df.rename(columns={df.columns[0]: "date"})

            # Convert Date to datetime
            if self.freq == 'daily':
                df["date"] = pd.to_datetime(
                    df["date"].astype(str),
                    format="%Y%m%d"
                )

            # Fama-French returns are in percent; convert to decimals
            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
            df[factor_cols] /= 100

            return df

        elif self.freq == 'monthly':
            response = requests.get(self.link, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]

                if not csv_files:
                    raise ValueError("No CSV found in ZIP")

                with z.open(csv_files[0]) as f:
                    text = f.read().decode("utf-8")

            # Keep only the monthly section
            monthly_text = text.split("Annual Factors:", 1)[0]

            df = pd.read_csv(
                io.StringIO(monthly_text),
                skiprows=3
            )

            df = df.rename(columns={df.columns[0]: "date"})


            # Convert Date to datetime
            df["date"] = (
                pd.to_datetime(df["date"].astype(str), format="%Y%m")
                + pd.offsets.BMonthEnd(0)
            )

            # Fama-French returns are in percent; convert to decimals
            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
            df[factor_cols] /= 100

            return df
    
        elif self.freq == 'annually':
            response = requests.get(self.link, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]

                if not csv_files:
                    raise ValueError("No CSV found in ZIP")

                with z.open(csv_files[0]) as f:
                    text = f.read().decode("utf-8")
        
            # Keep only the annual section
            annual_text = text.split("Annual Factors:", 1)[1]
        
            df = pd.read_csv(
                io.StringIO(annual_text),
                skiprows=1,
                skipfooter=2
            )
        
            df = df.rename(columns={df.columns[0]: "date"})
        
            # Convert Date to datetime
            df["date"] = (
                pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y")
                + pd.offsets.BYearEnd(0)
            )
        
            # Fama-French returns are in percent; convert to decimals
            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
            df[factor_cols] /= 100
        
            return df


    def get_data(self, start_date=None, end_date=None):
        """
        Return French historical factor data for the requested date range.

        Data is cached locally in Parquet files.

        The cache will only be updated when the cache becomes stale
        dictated by self.REFRESH_INTERVALS, so the cache data date range
        may not be the latest.
        
        This explicit design is chosen to balance
        between conserving API call and having enough data points
        for regression.

        Writing the cache is best-effort: if the cache folder is missing,
        read-only or otherwise unwritable (e.g. on a hosted app), a warning
        is logged and the freshly downloaded data is used directly.
        
        """

        
        cache_file = self._get_cache_file()

        # ------------------------------------------------------
        # Requested date range
        # ------------------------------------------------------

        requested_start = (
            pd.Timestamp("1900-01-01") if start_date is None else pd.Timestamp(start_date)
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
        
            df = self._get_full_history()

            if df.empty:
                return self._empty_dataframe()

            # Best-effort: a failed write must not fail the request.
            self._save_cache(df, cache_file)

            # Use the frame we just downloaded instead of re-reading
            # it from disk (same clean-up as _load_cache applies).
            cached = self._normalise(df)

        # ------------------------------------------------------
        # Cache is fresh -> use local data
        # ------------------------------------------------------

        else:
            cached = self._load_cache(cache_file)

        return self._filter_date_range(
            cached,
            requested_start,
            requested_end
        )


    def _empty_dataframe(self):
        """
        Return an empty DataFrame with the correct schema.
        """

        columns = ['date', *self.FACTOR_COLS]

        return pd.DataFrame(columns=columns)


    
    def _cache_is_stale(self, cache_file):
        if not cache_file.exists():
            return True

        cache_age = datetime.now() - datetime.fromtimestamp(
            cache_file.stat().st_mtime
        )

        return cache_age >= self.REFRESH_INTERVALS[self.freq]


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
    def _normalise(df):
        """
        Standard clean-up applied to factor data: normalised dates,
        no duplicate dates, sorted by date.
        """

        df = df.copy()

        df["date"] = (
            pd.to_datetime(df["date"])
            .dt.normalize()
        )

        return (
            df
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

    
    # ==========================================================
    # Cache
    # ==========================================================

    def _get_cache_file(self):
        """
        Return the cache path for this region / frequency.

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
                "Could not create French factor cache folder %s: %s",
                cache_dir,
                exc,
            )

        return cache_dir / (
            f"{self.region}_{self.freq}_.parquet"
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
                f"Unable to read French factor cache:\n"
                f"{cache_file}"
            ) from exc

        if cached.empty:
            return self._empty_dataframe()

        if "date" not in cached.columns:
            raise RuntimeError(
                f"French factor cache is missing the 'date' column:\n"
                f"{cache_file}"
            )

        return self._normalise(cached)

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
                "Could not write French factor cache %s: %s",
                cache_file,
                exc,
            )

            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass