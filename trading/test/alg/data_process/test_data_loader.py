from pathlib import Path

import pandas as pd
import pytest

import trading.src.alg.data_process.data_loader
from trading.cli.alg.config import ProjectPath
from trading.test.alg.test_fixtures import *


def test_data_loader(data_loader):
    """
    Test the DataLoader initialization and data fetching.
    """
    assert data_loader.df.shape[0] > 0, "DataFrame should not be empty"

    assert "close" in data_loader.df.columns, "DataFrame should contain 'close' column"
    assert "open" in data_loader.df.columns, "DataFrame should contain 'open'' column"
    assert "high" in data_loader.df.columns, "DataFrame should contain 'high' column"
    assert "low" in data_loader.df.columns, "DataFrame should contain 'low' column"
    assert (
        "volume" in data_loader.df.columns
    ), "DataFrame should contain 'volume' column"

    # Check MultiIndex levels
    assert "timestamp" in data_loader.df.index.names, "Index should contain 'timestamp'"
    assert "symbol" in data_loader.df.index.names, "Index should contain 'symbol'"

    # Check only one ticker
    assert (
        data_loader.df.index.get_level_values("symbol").nunique() == 1
    ), "DataFrame should contain only one ticker"
    assert (
        data_loader.df.index.get_level_values("symbol")[0] == "AAPL"
    ), "Ticker should be 'AAPL'"

    # Check timestamp monotonicity
    timestamps = data_loader.df.index.get_level_values("timestamp")
    assert timestamps.is_monotonic_increasing, "Timestamp should be in increasing order"

    # Check for NaNs
    assert (
        data_loader.df.isnull().sum().sum() == 0
    ), "DataFrame should not contain NaN values"


def test_data_loader_split(data_loader):
    """
    Test the DataLoader's train-test split functionality.
    """
    train_df, test_df = data_loader.get_train_test()

    assert len(train_df) > 0, "Training DataFrame should not be empty"
    assert len(test_df) > 0, "Testing DataFrame should not be empty"

    assert len(train_df) + len(test_df) == len(
        data_loader.df
    ), "Split should cover all data"

    train_timestamps = train_df.index.get_level_values("timestamp")
    test_timestamps = test_df.index.get_level_values("timestamp")
    assert (
        train_timestamps.max() < test_timestamps.min()
    ), "Training data should end before testing data starts"

    assert (
        train_df.index.get_level_values("symbol").nunique() == 1
    ), "Training DataFrame should contain only one ticker"
    assert (
        test_df.index.get_level_values("symbol").nunique() == 1
    ), "Testing DataFrame should contain only one ticker"


def test_multi_ticker(data_config, feature_config):
    from trading.src.alg.data_process.data_loader import DataLoader

    data_config.requests[0].kwargs["symbol_or_symbols"] = [
        "AAPL",
        "MSFT",
        "GOOGL",
    ]
    data_config.requests[0].dataset_name = "TEST_MULTI_TICKERS"
    feature_config.features[0].source = "TEST_MULTI_TICKERS"
    data_loader = DataLoader(data_config=data_config, feature_config=feature_config)

    assert data_loader.df.shape[0] > 0, "DataFrame should not be empty"
    symbols = data_loader.df.index.get_level_values("symbol")
    assert symbols.nunique() == 3, "DataFrame should contain three tickers"
    assert set(symbols.unique()) == {
        "AAPL",
        "MSFT",
        "GOOGL",
    }, "DataFrame should contain AAPL, MSFT, and GOOGL tickers"


def test_unit_period_pull(data_config, feature_config):
    from trading.src.alg.data_process.data_loader import DataLoader

    data_config.time_step_unit = "Min"
    data_config.time_step_period = 5
    data_config.start_date = "2025-7-8 08:00:00"
    data_config.end_date = "2025-7-8 16:00:00"
    data_config.requests[0].dataset_name = "TEST_TIME_STEP"
    feature_config.features[0].source = "TEST_TIME_STEP"
    data_loader = DataLoader(data_config=data_config, feature_config=feature_config)

    assert data_loader.df.shape[0] > 0, "DataFrame should not be empty"
    timestamps = data_loader.df.index.get_level_values("timestamp")
    assert (
        timestamps.minute % 5 == 0
    ).all(), "All timestamps should be on 5-minute intervals"
    assert (
        pd.Series(timestamps).dt.date.nunique() == 1
    ), "DataFrame should contain data for only 1 day"
    assert (
        len(data_loader.df) == 97
    ), "DataFrame should contain 97 rows for 5-minute intervals in one day (inclusive of end)"
