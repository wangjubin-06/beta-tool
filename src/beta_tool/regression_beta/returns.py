import numpy as np
import pandas as pd

def log_returns(data: pd.DataFrame, header_name = "adjClose") -> pd.DataFrame:
    df = data.copy()
    df["log-returns"] = np.log(df[header_name] / df[header_name].shift(1))
    df = df.dropna().reset_index(drop=True)
    
    return df

def simple_returns(data: pd.DataFrame, header_name = "adjClose") -> pd.DataFrame:
    df = data.copy()
    df["simple-returns"] = (df[header_name] / df[header_name].shift(1)) - 1
    df = df.dropna().reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    pass

