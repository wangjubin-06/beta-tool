from lse import LSE
import os
import numpy as np
import pandas as pd

from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta



def generate_risk_free_rate_df(start: str, end:str = None, return_type:str = 'log') -> pd.DataFrame:

    if return_type not in {'simple','log'}:
        raise ValueError("select proper return_type: log or simple")

    try:
        # %Y = 4-digit year, %m = 2-digit month, %d = 2-digit day
        datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Start date '{start}' does not match format 'YYYY-MM-DD'")

    if end is not None:
        try:
            # %Y = 4-digit year, %m = 2-digit month, %d = 2-digit day
            datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"End date '{end}' does not match format 'YYYY-MM-DD'")
    
    client = LSE(api_key=os.environ.get('LSE_API_KEY')) #User has to sign up for an account at https://londonstrategicedge.com/ and save their own api key in their system's environment variables under the name 'LSE_API_KEY'

    if end is None:
        risk_free_rate_list = client.series(symbol="US1M", start=start)
    else:
        risk_free_rate_list = client.series(symbol="US1M", start=start, end=end)
    # this is a list of dict objects
    # each dict object is in the format {
    # {
    #   'symbol' : 'US1M'
    #   'date' : 'yyyy-MM-dd'
    #   'value' : yield of US1M on that day in float type
    # }


    df_dict = {
        'timestamp':[],
        f'daily-rf-{return_type}-returns':[]
    }

    for dict_obj in risk_free_rate_list:
        df_dict['timestamp'].append(dict_obj['date'])
        if return_type == 'log':
            daily_rf_log = np.log1p((dict_obj['value']/100) / 360)
            df_dict[f'daily-rf-{return_type}-returns'].append(daily_rf_log)
        else:
            daily_rf = (dict_obj['value']/100) / 360
            df_dict[f'daily-rf-{return_type}-returns'].append(daily_rf)

    

    df = pd.DataFrame(df_dict)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    ret_df = df.copy()

    ret_df["timestamp"] = pd.to_datetime(ret_df["timestamp"], errors="coerce")

    ret_df = (
        ret_df.dropna(subset=["timestamp", f'daily-rf-{return_type}-returns'])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    return ret_df


if __name__ == '__main__':
    client = LSE(api_key=os.environ.get('LSE_API_KEY'))
    risk_free_rate_list = client.series(symbol="US1M")
    df = generate_risk_free_rate_df(start="2026-07-01", return_type="log")
    print(df.head())

