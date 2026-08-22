from beta_tool.regression_beta.returns import log_returns, simple_returns
import pandas as pd
import numpy as np

def test_log_returns():
    data = pd.DataFrame({
        "adjClose": [100, 110, 121]
    })


    result = log_returns(data)

    expected = np.log(np.array([110, 121]) / np.array([100, 110]))

    np.testing.assert_allclose(
        result["log-returns"].to_numpy(),
        expected
    )


def test_simple_returns():
    data = pd.DataFrame({
        "adjClose": [100, 110, 121]
    })


    result = simple_returns(data)

    expected = np.array([0.10, 0.10])

    np.testing.assert_allclose(
        result["simple-returns"].to_numpy(),
        expected
    )


def test_log_returns_drops_first_row():
    data = pd.DataFrame({
        "adjClose": [100, 110, 121]
    })

    
    result = log_returns(data)

    assert len(result) == 2
    assert "log-returns" in result.columns


def test_simple_returns_drops_first_row():
    data = pd.DataFrame({
        "adjClose": [100, 110, 121]
    })


    result = simple_returns(data)

    assert len(result) == 2
    assert "simple-returns" in result.columns


def test_log_returns_does_not_modify_original_data():
    data = pd.DataFrame({
        "adjClose": [100, 110, 121]
    })


    original = data.copy()

    log_returns(data)

    pd.testing.assert_frame_equal(data, original)


def test_simple_returns_does_not_modify_original_data():
    data = pd.DataFrame({
        "adjClose": [100, 110, 121]
    })


    original = data.copy()

    simple_returns(data)

    pd.testing.assert_frame_equal(data, original)


def test_custom_header_name():
    data = pd.DataFrame({
        "price": [100, 110, 121]
    })


    result = simple_returns(data, header_name="price")

    expected = np.array([0.10, 0.10])

    np.testing.assert_allclose(
        result["simple-returns"].to_numpy(),
        expected
    )

def test_log_returns_missing_data():
    data = pd.DataFrame({
        "adjClose": [100, None, 110, None, 121]
    })
    
    result = log_returns(data)
    
    expected = np.log(np.array([110, 121]) / np.array([100, 110]))
    
    np.testing.assert_allclose(
        result["log-returns"].to_numpy(),
        expected
    )

def test_simple_returns_missing_data():
    data = pd.DataFrame({
        "adjClose": [100, None, 110, None, 121]
    })
    
    result = simple_returns(data)
    
    expected = np.array([110, 121]) / np.array([100, 110])
    
    np.testing.assert_allclose(
        result["simple-returns"].to_numpy(),
        expected
    )