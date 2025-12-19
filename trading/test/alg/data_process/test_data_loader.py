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


# ! TODO: add test for delisted tickers (alpca does not provide this endpoint currently) Need to train and test with delisted tickers
# ! to prevent survivorship bias


def test_incomplete_multi_ticker(incomplete_multi_ticker_data_loader):
    data_loader = incomplete_multi_ticker_data_loader

    assert data_loader.df.shape[0] > 0, "DataFrame should not be empty"
    symbols = data_loader.df.index.get_level_values("symbol")
    assert symbols.nunique() == 4, "DataFrame should contain four tickers"
    assert set(symbols.unique()) == {
        "AAPL",
        "MSFT",
        "GOOGL",
        "HG",  # IPO'd on November 10th 2023
    }, "DataFrame should contain AAPL, MSFT, GOOGL, and HG tickers"

    df = data_loader.df.drop("timestamp", axis=1)
    hg_df = df.xs("HG", level="symbol")
    aapl_df = df.xs("AAPL", level="symbol")
    msft_df = df.xs("MSFT", level="symbol")
    googl_df = df.xs("GOOGL", level="symbol")

    assert (
        hg_df.shape == aapl_df.shape == msft_df.shape == googl_df.shape
    ), "All tickers should have the same number of rows"

    assert hg_df.iloc[0].equals(
        hg_df.iloc[100]
    ), "HG data should be backfilled correctly"
    assert not aapl_df.iloc[0].equals(
        aapl_df.iloc[1]
    ), "AAPL data should not be backfilled"
    assert not msft_df.iloc[0].equals(
        msft_df.iloc[1]
    ), "MSFT data should not be backfilled"
    assert not googl_df.iloc[0].equals(
        googl_df.iloc[1]
    ), "GOOGL data should not be backfilled"


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


def test_multi_ticker(multi_data_loader):
    data_loader = multi_data_loader

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


def test_cache_fetches_missing_ranges_and_updates(monkeypatch, tmp_path):
    """
    Verify partial-cache scenarios fetch only missing date ranges, merge, save, and return filtered data.
    """

    import types

    from alpaca.data.timeframe import TimeFrameUnit

    from trading.cli.alg.config import DataRequests, DataSourceType
    from trading.src.alg.data_process import data_loader as dl
    from trading.src.alg.data_process.data_loader import AlpacaDataLoader

    def make_df(symbols, start, end):
        dates = pd.date_range(start=start, end=end, freq="D")
        idx = pd.MultiIndex.from_product(
            [dates, symbols], names=["timestamp", "symbol"]
        )
        data = {
            "open": range(1, len(idx) + 1),
            "high": range(2, len(idx) + 2),
            "low": range(0, len(idx)),
            "close": range(1, len(idx) + 1),
            "volume": [100] * len(idx),
        }
        return pd.DataFrame(data, index=idx)

    class _Secret:
        def __init__(self, val):
            self.val = val

        def get_secret_value(self):
            return self.val

    class _UserCache:
        def load(self):
            return types.SimpleNamespace(
                alpaca_api_key=_Secret("key"), alpaca_api_secret=_Secret("secret")
            )

    requests_made = []

    class _MockBars:
        def __init__(self, df):
            self.df = df

    class _StockHistoricalDataClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_stock_bars(self, request_params):
            start = pd.to_datetime(request_params.start)
            end = pd.to_datetime(request_params.end)
            symbols = request_params.symbol_or_symbols
            requests_made.append((start, end, tuple(symbols)))
            return _MockBars(make_df(symbols, start, end))

    monkeypatch.setattr(dl.user_cache, "UserCache", _UserCache)
    monkeypatch.setattr(dl, "StockHistoricalDataClient", _StockHistoricalDataClient)

    cache_path = tmp_path
    dataset_name = "CACHE_TEST"
    cache_file = cache_path / f"{dataset_name}.parquet"

    cache_df = make_df(["AAPL"], "2023-01-03", "2023-01-05")
    cache_df.to_parquet(cache_file)

    request = DataRequests(
        dataset_name=dataset_name,
        source=DataSourceType.ALPACA,
        endpoint="StockBarRequest",
        kwargs={"symbol_or_symbols": ["AAPL"], "adjustment": "split"},
    )

    loader = AlpacaDataLoader()
    result = loader.get_data(
        fetch_data=False,
        request=request,
        df=pd.DataFrame(),
        cache_path=str(cache_path),
        start_date="2023-01-01",
        end_date="2023-01-06",
        time_step_unit=TimeFrameUnit.Day,
        cache_enabled=True,
        time_step_period=1,
    )

    # ensure two segments were fetched: before cache start and after cache end
    assert len(requests_made) == 2
    assert requests_made[0][0] == pd.Timestamp("2023-01-01")
    assert requests_made[0][1] == pd.Timestamp("2023-01-03")
    assert requests_made[1][0] == pd.Timestamp("2023-01-05")
    assert requests_made[1][1] == pd.Timestamp("2023-01-06")

    # cache should be updated to cover full requested range
    updated_cache = pd.read_parquet(cache_file)
    assert updated_cache.index.get_level_values("timestamp").min() == pd.Timestamp(
        "2023-01-01"
    )
    assert updated_cache.index.get_level_values("timestamp").max() == pd.Timestamp(
        "2023-01-06"
    )
    assert len(updated_cache) == 6  # one symbol, 6 days inclusive

    # returned data should match requested window and be sorted without duplicates
    timestamps = result.index.get_level_values("timestamp")
    assert timestamps.min() == pd.Timestamp("2023-01-01")
    assert timestamps.max() == pd.Timestamp("2023-01-06")
    assert timestamps.is_monotonic_increasing
    assert result.index.is_unique
    assert len(result) == 6
