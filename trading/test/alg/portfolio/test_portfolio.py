import logging

import pandas as pd
import pytest

from trading.src.alg.portfolio.portfolio import Portfolio


@pytest.fixture
def single_tic_data():
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="D"),
        "symbol": ["AAPL"] * 5,
        "close": [10, 10, 20, 10, 155],
        "size": [10, 0, -10, 0, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dates():
    return pd.date_range(start="2023-01-01", periods=5, freq="D")


@pytest.fixture
def multi_tic_data(dates):
    tics = ["AAPL", "MSFT", "NVDA"]
    close_prices = {
        "AAPL": [10, 12, 14, 16, 18],
        "MSFT": [20, 22, 24, 26, 28],
        "NVDA": [30, 32, 34, 36, 38],
    }
    sizes = {
        "AAPL": [10, -5, -5, 0, 0],
        "MSFT": [10, 10, 0, -5, 0],
        "NVDA": [0, 0, 10, 0, -5],
    }
    records = []
    for i, date in enumerate(dates):
        for tic in tics:
            records.append(
                {
                    "timestamp": date,
                    "symbol": tic,
                    "close": close_prices[tic][i],
                    "size": sizes[tic][i],
                }
            )
    df = pd.DataFrame(records).set_index(["timestamp", "symbol"])
    return df


def test_portfolio_multi(multi_tic_data, dates):
    pf = Portfolio(initial_cash=1000)
    assert pf.cash == 1000, "Initial cash should be set to 1000"
    assert pf.total_value == 1000, "Total value should be equal to initial cash"
    assert pf.nav == 0, "NAV should be initialized to 0"

    for i, date in enumerate(dates):
        pf.update_position_batch(multi_tic_data.loc[[date]])

    assert pf.as_vbt_pf() is not None, "VectorBT portfolio should be created"
    assert pf.total_value > 1000, "Total value should increase with positive trades"
    assert pf.nav > 0, "NAV should be greater than 0 after trades"
