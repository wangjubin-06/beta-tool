import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def price_series_plot(data: pd.DataFrame) -> plt.plot:
    df = data.copy()

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["timestamp"],
        df["adjusted_close"],
        label="adjusted-close price"
    )

    plt.title(
        f"Price series of {df['symbol'].loc[1]}",
        fontsize=16,
        pad = 15
        )
    
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


if __name__ == "__main__":
    import data
    apple = data.AssetData(ticker="aapl",interval="daily",start_date = "2013-01-01", end_date = "2021-01-01")
    data = apple.get_prices()

    price_series_plot(data=data)