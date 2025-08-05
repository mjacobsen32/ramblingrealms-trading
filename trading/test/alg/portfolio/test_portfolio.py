import logging

import numpy as np
import pandas as pd
import pytest

from trading.src.alg.portfolio.portfolio import Portfolio
from trading.test.alg.test_fixtures import *


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
def expected_states():
    expected_states = [
        [700, 10, 10, 0],
        [540, 5, 20, 0],
        [270, 0, 20, 10],
        [400, 0, 15, 10],
        [590, 0, 15, 5],
    ]
    return expected_states


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


def test_portfolio_data_set(data_loader, portfolio_config):
    portfolio_config.initial_cash = 1000
    pf = Portfolio(cfg=portfolio_config, stock_dimension=1)
    np.random.seed(42)  # For reproducible results
    data = data_loader.get_train_test()[0].copy()
    data["size"] = np.random.choice([-1, 0, 1], size=len(data))
    for date in data.index:
        pf.update_position_batch(data.loc[[date]])
    vbt = pf.as_vbt_pf()


def test_portfolio_multi_data_set(multi_data_loader, portfolio_config):
    portfolio_config.initial_cash = 1000
    pf = Portfolio(cfg=portfolio_config, stock_dimension=1)
    np.random.seed(42)  # For reproducible results
    data = multi_data_loader.get_train_test()[0].copy()
    data["size"] = np.random.choice([-1, 0, 1], size=len(data))
    for date in data.index.get_level_values("timestamp").unique():
        pf.update_position_batch(data.loc[[date]])
    vbt = pf.as_vbt_pf()
    prof = vbt.total_profit()
    assert len(vbt.total_profit(group_by=False)) == 3
    assert (
        prof == vbt.total_profit(group_by=False).sum()
    ), "Total profit should match sum of individual profits"


def test_portfolio_multi(multi_tic_data, dates, expected_states, portfolio_config):
    portfolio_config.initial_cash = 1000
    pf = Portfolio(cfg=portfolio_config, stock_dimension=3)
    assert pf.initial_cash == 1000, "Initial cash should be set to 1000"
    assert pf.cash == 1000, "Initial cash should be set to 1000"
    assert pf.total_value == 1000, "Total value should be equal to initial cash"
    assert pf.nav == 0, "NAV should be initialized to 0"
    assert pf.vbt_pf == None, "VectorBT portfolio should be None initially"
    assert pf.stock_dimension == 3, "Stock dimension should be set to 1"

    np.testing.assert_array_equal(
        pf.state(), np.array([1000, 0, 0, 0])
    ), "Initial state should be [cash, positions]"

    for i, date in enumerate(dates):
        pf.update_position_batch(multi_tic_data.loc[[date]])
        expected = np.array(expected_states[i])
        actual = pf.state(date)
        np.testing.assert_array_equal(
            actual, expected, f"State at date {date} should match expected values"
        )

    assert pf.total_value > 1000, "Total value should increase with positive trades"
    assert pf.nav > 0, "NAV should be greater than 0 after trades"

    vbt = pf.as_vbt_pf()
    assert vbt is not None, "VectorBT portfolio should be created"
    assert pf.vbt_pf == vbt, "VectorBT portfolio should be stored in the portfolio"

    assert (
        len(pf.orders()) == 8
    ), "There should be 8 orders in the portfolio (non-zero sizes)"

    assert len(vbt.total_profit(group_by=False)) == 3
    assert (
        vbt.total_profit() == vbt.total_profit(group_by=False).sum()
    ), "Total profit should match sum of individual profits"


@pytest.fixture
def constraints_portfolio_config():
    """
    Fixture to create a simple portfolio configuration.
    """
    from trading.cli.alg.config import PortfolioConfig, SellMode, TradeMode

    return PortfolioConfig(
        initial_cash=1_000,
        hmax=100,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        max_positions=1,
        trade_mode=TradeMode.CONTINUOUS,
        sell_mode=SellMode.CONTINUOUS,
        trade_limit_percent=0.1,
    )


def test_cash_limit(constraints_portfolio_config):
    cfg = constraints_portfolio_config
    cfg.initial_cash = 1000
    cfg.hmax = 1000
    cfg.trade_limit_percent = 1.0
    pf = Portfolio(cfg=cfg, stock_dimension=2)
    df = pd.DataFrame(
        {
            "timestamp": [
                d
                for d in pd.date_range(start="2023-01-01", periods=3, freq="D")
                for _ in range(2)
            ],
            "symbol": ["AAPL", "NVDA"] * 3,
            "close": [10, 10, 10, 10, 10, 10],
            "size": [101, 0, -101, 0, 51, 51],
        }
    ).set_index(["timestamp", "symbol"])

    ret = pf.step(df.iloc[0:2]["close"], df.iloc[0:2])
    np.testing.assert_array_equal(
        ret["scaled_actions"],
        [100.0, 0.0],
        "Action should be scaled to max position size",
    )

    ret = pf.step(df.iloc[2:4]["close"], df.iloc[2:4])
    np.testing.assert_array_equal(
        ret["scaled_actions"],
        [-100.0, 0.0],
        "Action should be scaled to max position size",
    )

    ret = pf.step(df.iloc[4:6]["close"], df.iloc[4:6])
    np.testing.assert_array_equal(
        ret["scaled_actions"],
        [50.0, 50.0],
        "Action should be scaled to max shares with remaining cash",
    )
