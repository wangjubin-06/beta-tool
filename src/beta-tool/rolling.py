import regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def historical_rolling_beta(beta_obj, observation_window:int = 60):
    """
    
    this function takes in a Beta object and calculates the beta over the previous observation window (in whole number of days) for every day, inclusive
    
    """
    if observation_window <= 0:
        raise ValueError(
            f"observation_window must be positive, got {observation_window}."
        )

    df = beta_obj.merged_returns_series.sort_values("timestamp").reset_index(drop=True)

    if len(df) < observation_window:
        raise ValueError(
            f"Insufficient data for rolling beta: "
            f"observation_window={observation_window}, "
            f"but only {len(df)} observations are available."
        )

    return_type = beta_obj.return_type
    asset_1_col = f"{return_type}-returns_1"
    asset_2_col = f"{return_type}-returns_2"


    rolling_beta_results = {}

    for i in range(observation_window - 1, len(df)):
        window = df.iloc[i - observation_window + 1:i + 1].copy()

        asset_1_returns = window[
            ["timestamp", asset_1_col]
        ]

        asset_2_returns = window[
            ["timestamp", asset_2_col]
        ]

        results = regression.OLSRegression(
            asset1=asset_1_returns,
            asset2=asset_2_returns,
            return_type=return_type
        ).ols()

        rolling_beta_results[window["timestamp"].iloc[-1]] = (
            results.params[asset_2_col]
        )

    

    return_df = pd.DataFrame(
        rolling_beta_results.items(),
        columns=["timestamp", "beta"]
    )


    return_df.attrs["title"] = f"Historical {observation_window}-trading-days rolling beta of {beta_obj.asset1} {beta_obj.return_type}-returns against {beta_obj.asset2} {beta_obj.return_type}-returns"

    return_df.attrs["period"] = f"from {str(return_df['timestamp'].iloc[0])[:10]} to {str(return_df['timestamp'].iloc[-1])[:10]}"

    return return_df


def historical_rolling_beta_plot(rolling_df: pd.DataFrame, ax = None):
    df = rolling_df.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(14,6))

    ax.plot(
        df["timestamp"],
        df["beta"],
    )

    ax.set_title(
        f"{df.attrs['title']}\n{df.attrs['period']}",
        fontsize=16,
        pad=15
    )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)

    ax.grid(alpha=0.2)

    return ax

        