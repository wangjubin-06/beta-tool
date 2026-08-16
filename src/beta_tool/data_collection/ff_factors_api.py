import requests
import pandas as pd
import os
from pathlib import Path
import io
import zipfile
from datetime import datetime, timedelta


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
            response = requests.get(self.link)
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
            response = requests.get(self.link)
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
            response = requests.get(self.link)
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
        CACHE_DIR = Path("../../data/french")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_file = CACHE_DIR / f"{self.region}_{self.freq}.parquet"

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
            df = self._get_full_history()

            # df["date"] = pd.to_datetime(df["date"])

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
