import numpy as np
import pandas as pd

def log_returns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["log-returns"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()
    df = df.drop(columns=['close'])
    return df

def simple_returns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["simple-returns"] = (df["close"] / df["close"].shift(1)) - 1
    df = df.dropna()
    df = df.drop(columns=['close'])
    return df