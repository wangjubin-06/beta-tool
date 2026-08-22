from beta_tool.regression_beta.regression import OLSRegression, MultiFactorRegression
import numpy as np
import pandas as pd
import pytest


def test_ols_regression_merges_overlapping_dates():
    asset1 = pd.DataFrame({
        "date": pd.to_datetime([
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        ]),
        "log-returns": [0.01, 0.02, 0.03],
    })

    
    asset2 = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ]),
        "log-returns": [0.10, 0.20, 0.30],
    })

    regression = OLSRegression(asset1, asset2)

    assert len(regression.merged_return_series) == 2

    assert list(regression.merged_return_series["date"]) == list(
        pd.to_datetime([
            "2024-01-02",
            "2024-01-03",
        ])
    )


def test_ols_regression_assigns_x_and_y_correctly():
    asset1 = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "log-returns": [0.02, 0.04],
    })

    
    asset2 = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "log-returns": [0.01, 0.02],
    })

    regression = OLSRegression(asset1, asset2)

    np.testing.assert_array_equal(
        regression.y.to_numpy(),
        [0.02, 0.04]
    )

    np.testing.assert_array_equal(
        regression.x.to_numpy(),
        [0.01, 0.02]
    )


def test_ols_regression_drops_missing_returns():
    asset1 = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ]),
        "log-returns": [0.01, np.nan, 0.03],
    })

    
    asset2 = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ]),
        "log-returns": [0.10, 0.20, np.nan],
    })

    regression = OLSRegression(asset1, asset2)

    assert len(regression.merged_return_series) == 1

    assert regression.merged_return_series["date"].iloc[0] == pd.Timestamp(
        "2024-01-01"
    )


def test_ols_regression_produces_expected_coefficients():
    asset2 = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "log-returns": [1, 2, 3, 4, 5],
    })

    
    # y = 2x + 1
    asset1 = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "log-returns": [3, 5, 7, 9, 11],
    })

    regression = OLSRegression(asset1, asset2)

    model = regression.ols()

    assert model.params["const"] == pytest.approx(1.0)
    assert model.params["log-returns_2"] == pytest.approx(2.0)


def test_multifactor_requires_at_least_one_factor():
    asset1 = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "log-returns": [1, 2, 3],
    })


    with pytest.raises(ValueError, match="at least one factor"):
        MultiFactorRegression(asset1, {})


def test_multifactor_merges_all_assets():
    dates = pd.date_range("2024-01-01", periods=3)

    
    asset1 = pd.DataFrame({
        "date": dates,
        "log-returns": [3, 5, 7],
    })

    factor1 = pd.DataFrame({
        "date": dates,
        "log-returns": [1, 2, 3],
    })

    factor2 = pd.DataFrame({
        "date": dates,
        "log-returns": [2, 4, 6],
    })

    regression = MultiFactorRegression(
        asset1,
        {
            "market": factor1,
            "factor2": factor2,
        }
    )

    assert regression.assets_names == ["market", "factor2"]

    assert list(regression.x.columns) == [
        "market",
        "factor2",
    ]

    np.testing.assert_array_equal(
        regression.y.to_numpy(),
        [3, 5, 7]
    )


def test_multifactor_requires_return_column():
    asset1 = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=2),
        "log-returns": [1, 2],
    })

    
    factor = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=2),
        "price": [100, 101],
    })

    with pytest.raises(ValueError, match="missing column"):
        MultiFactorRegression(
            asset1,
            {"market": factor}
        )


def test_multifactor_no_overlapping_dates():
    asset1 = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "log-returns": [1, 2],
    })

    
    factor = pd.DataFrame({
        "date": pd.to_datetime(["2024-02-01", "2024-02-02"]),
        "log-returns": [3, 4],
    })

    with pytest.raises(
        ValueError,
        match="no overlapping timestamps"
    ):
        MultiFactorRegression(
            asset1,
            {"market": factor}
        )


def test_multifactor_produces_expected_coefficients():
    dates = pd.date_range("2024-01-01", periods=5)

    
    market = np.array([1, 2, 3, 4, 5])
    value = np.array([5, 1, 4, 2, 6])

    # y = 1 + 2 * market + 3 * value
    y = 1 + 2 * market + 3 * value

    asset1 = pd.DataFrame({
        "date": dates,
        "log-returns": y,
    })

    market_df = pd.DataFrame({
        "date": dates,
        "log-returns": market,
    })

    value_df = pd.DataFrame({
        "date": dates,
        "log-returns": value,
    })

    regression = MultiFactorRegression(
        asset1,
        {
            "market": market_df,
            "value": value_df,
        }
    )

    model = regression.ols()

    assert model.params["const"] == pytest.approx(1.0)
    assert model.params["market"] == pytest.approx(2.0)
    assert model.params["value"] == pytest.approx(3.0)
