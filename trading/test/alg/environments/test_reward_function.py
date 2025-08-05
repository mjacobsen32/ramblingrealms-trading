import logging

import pandas as pd
import pytest

from trading.cli.alg.config import PortfolioConfig, RewardConfig
from trading.src.alg.environments.reward_functions.basic_profit_max import (
    BasicProfitMax,
)
from trading.src.alg.portfolio.portfolio import Portfolio
from trading.test.alg.test_fixtures import *


@pytest.fixture
def basic_reward_function():
    basic_reward_function = BasicProfitMax(
        cfg=RewardConfig(type="basic_profit_max", reward_scaling=1000.0)
    )
    return basic_reward_function


@pytest.fixture
def mild_profitable_portfolio(portfolio_config):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.to_datetime("2023-01-01"),
                periods=5,
                freq="D",
            ),
            "symbol": ["AAPL"] * 5,
            "close": [10, 11, 12, 13, 14],
            "size": [30, 20, 10, -40, -20],
        }
    ).set_index(["timestamp", "symbol"])
    portfolio_config.initial_cash = 5_000
    pf = Portfolio(cfg=portfolio_config)
    pf.update_position_batch(df)
    return pf


@pytest.fixture
def mild_negative_portfolio(portfolio_config):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00"), periods=5, freq="D"
            ),
            "symbol": ["AAPL"] * 5,
            "close": [14, 13, 12, 11, 10],
            "size": [30, 20, 10, -40, -20],
        }
    ).set_index(["timestamp", "symbol"])
    portfolio_config.initial_cash = 5_000
    pf = Portfolio(cfg=portfolio_config)
    pf.update_position_batch(df)
    return pf


@pytest.fixture
def strong_profitable_portfolio(portfolio_config):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00"), periods=5, freq="D"
            ),
            "symbol": ["AAPL"] * 5,
            "close": [10, 50, 100, 150, 200],
            "size": [30, 20, 10, -40, -20],
        }
    ).set_index(["timestamp", "symbol"])
    portfolio_config.initial_cash = 5_000
    pf = Portfolio(cfg=portfolio_config)
    pf.update_position_batch(df)
    return pf


@pytest.fixture
def strong_negative_portfolio(portfolio_config):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00"),
                end=pd.to_datetime("2023-01-05 00:00:00"),
            ),
            "symbol": ["AAPL"] * 5,
            "close": [200, 150, 100, 50, 10],
            "size": [30, 20, 10, -40, -20],
        }
    ).set_index(["timestamp", "symbol"])
    portfolio_config.initial_cash = 5_000
    pf = Portfolio(cfg=portfolio_config)
    pf.update_position_batch(df)
    return pf


def test_basic_profit_max(
    basic_reward_function,
    strong_profitable_portfolio,
    strong_negative_portfolio,
    mild_profitable_portfolio,
    mild_negative_portfolio,
):
    reward_function = basic_reward_function
    strong_profit = reward_function.compute_reward(
        strong_profitable_portfolio, "2023-01-05"
    )
    reward_function.reset()
    mild_profit = reward_function.compute_reward(
        mild_profitable_portfolio, "2023-01-05"
    )
    reward_function.reset()
    strong_negative = reward_function.compute_reward(
        strong_negative_portfolio, "2023-01-05"
    )
    reward_function.reset()
    mild_negative = reward_function.compute_reward(
        mild_negative_portfolio, "2023-01-05"
    )
    print(
        f"Strong Profit: {strong_profit}, Mild Profit: {mild_profit}, "
        f"Strong Negative: {strong_negative}, Mild Negative: {mild_negative}"
    )
    print(strong_profitable_portfolio)
    print(strong_negative_portfolio)
    print(mild_profitable_portfolio)
    print(mild_negative_portfolio)
    assert strong_profit > 0.9
    assert mild_profit > 0.01
    assert strong_negative < -0.9
    assert mild_negative < -0.01
    assert mild_profit < strong_profit
    assert mild_negative > strong_negative
