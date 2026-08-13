import numpy as np
import pandas as pd

def log_returns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["log-returns"] = np.log(df["adjusted_close"] / df["adjusted_close"].shift(1))
    df = df.dropna().reset_index(drop=True)
    
    return df

def simple_returns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["simple-returns"] = (df["adjusted_close"] / df["adjusted_close"].shift(1)) - 1
    df = df.dropna().reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    import data
    import os
    from lse import LSE
    client = LSE(api_key=os.environ.get('LSE_API_KEY'))

    apple = data.AssetData(ticker= "aapl", interval="daily", start_date = "2020-08-20", end_date   = "2020-09-10") 
    data = apple.get_prices()

    returns = log_returns(data)

