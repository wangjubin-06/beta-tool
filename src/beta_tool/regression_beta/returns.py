import numpy as np
import pandas as pd

def log_returns(data: pd.DataFrame, header_name = "adjClose") -> pd.DataFrame:
    
    if data.empty:
        raise ValueError("Price series dataframe is empty!")
    
    if not header_name in data.columns:
        raise KeyError(f"No column named {header_name} in dataframe provided for return calculation!")
    
    
    df = data.copy()
    
    df = df.dropna(subset=[header_name])
    
    df["log-returns"] = np.log(df[header_name] / df[header_name].shift(1))
    
    df = df.dropna().reset_index(drop=True)

    ret_df = df[['date', 'log-returns']]
    
    return ret_df


def simple_returns(data: pd.DataFrame, header_name = "adjClose") -> pd.DataFrame:
    
    if data.empty:
        raise ValueError("Price series dataframe is empty!")
    
    if not header_name in data.columns:
        raise KeyError(f"No column named {header_name} in dataframe provided for return calculation!")
    
    
    df = data.copy()
    
    df = df.dropna(subset=[header_name])
    
    df["simple-returns"] = (df[header_name] / df[header_name].shift(1)) - 1
    
    df = df.dropna().reset_index(drop=True)

    ret_df = df[['date', 'simple-returns']]
    
    return ret_df

