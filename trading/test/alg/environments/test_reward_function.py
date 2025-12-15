import numpy as np
import pandas as pd
import pytest

from trading.cli.alg.config import RewardConfig
from trading.src.alg.environments.reward_functions.basic_profit_max import (
    BasicProfitMax,
)
from trading.src.alg.environments.reward_functions.fast_profit_reward import (
    FastProfitReward,
    SimpleMomentumReward,
)
from trading.src.alg.environments.reward_functions.reward_function_factory import (
    reward_factory_method,
)
from trading.src.portfolio.portfolio import Portfolio
from trading.test.alg.test_fixtures import *


@pytest.fixture
def basic_reward_function():
    basic_reward_function = BasicProfitMax(
        cfg=RewardConfig(type="basic_profit_max", reward_scaling=1000.0),
        initial_state=np.array([5000, 0.0]),
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
            "price": [10, 11, 12, 13, 14],
            "size": [30, 20, 10, -40, -20],
            "returns": [0.0, 0.1, 0.2, 0.3, 0.4],
        }
    ).set_index(["timestamp", "symbol"])
    df["close"] = df["price"]
    df["timestamp"] = df.index.get_level_values("timestamp").unique()
    df["profit"] = 0.0
    portfolio_config.initial_cash = 5_000
    pf = Portfolio(cfg=portfolio_config, symbols=["AAPL"])
    for date in df.index.get_level_values("timestamp").unique():
        pf.update_position_batch(df.loc[date])
    return pf


@pytest.fixture
def mild_negative_portfolio(portfolio_config):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00"), periods=5, freq="D"
            ),
            "symbol": ["AAPL"] * 5,
            "price": [14, 13, 12, 11, 10],
            "size": [30, 20, 10, -40, -20],
            "returns": [0.0, -0.25, -0.33, -0.5, -0.8],
        }
    ).set_index(["timestamp", "symbol"])
    df["close"] = df["price"]
    df["timestamp"] = df.index.get_level_values("timestamp").unique()
    df["profit"] = 0.0
    portfolio_config.initial_cash = 5_000
    pf = Portfolio(cfg=portfolio_config, symbols=["AAPL"])
    for date in df.index.get_level_values("timestamp").unique():
        pf.update_position_batch(df.loc[date])
    return pf


@pytest.fixture
def strong_profitable_data():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00"), periods=5, freq="D"
            ),
            "symbol": ["AAPL"] * 5,
            "price": [10, 50, 100, 150, 200],
            "size": [30, 20, 10, -40, -20],
            "returns": [0.0, -0.20, 2.0, 1.0, 1.0],
        }
    ).set_index(["timestamp", "symbol"])
    df["close"] = df["price"]
    df["timestamp"] = df.index.get_level_values("timestamp").unique()
    df["profit"] = 0.0
    return df


@pytest.fixture
def strong_profitable_portfolio(portfolio_config, strong_profitable_data):
    portfolio_config.initial_cash = 5_000
    pf = Portfolio(cfg=portfolio_config, symbols=["AAPL"])
    for date in strong_profitable_data.index.get_level_values("timestamp").unique():
        pf.update_position_batch(strong_profitable_data.loc[date])
    return pf


@pytest.fixture
def strong_negative_data():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00"),
                end=pd.to_datetime("2023-01-05 00:00:00"),
            ),
            "symbol": ["AAPL"] * 5,
            "price": [200, 150, 100, 50, 10],
            "size": [30, 20, 10, -40, -20],
            "returns": [0.0, -0.25, -0.33, -0.5, -0.8],
        }
    ).set_index(["timestamp", "symbol"])
    df["close"] = df["price"]
    df["timestamp"] = df.index.get_level_values("timestamp").unique()
    df["profit"] = 0.0
    return df


@pytest.fixture
def strong_negative_portfolio(portfolio_config, strong_negative_data):
    portfolio_config.initial_cash = 5_000
    pf = Portfolio(cfg=portfolio_config, symbols=["AAPL"])
    for date in strong_negative_data.index.get_level_values("timestamp").unique():
        pf.update_position_batch(strong_negative_data.loc[date])
    return pf


def test_basic_profit_max(
    basic_reward_function,
    strong_negative_portfolio,
    strong_profitable_portfolio,
    mild_profitable_portfolio,
    mild_negative_portfolio,
):
    rf = basic_reward_function
    strong_profit = rf.compute_reward(strong_profitable_portfolio, [], 0.0)
    rf.reset()
    mild_profit = rf.compute_reward(mild_profitable_portfolio, [], 0.0)
    rf.reset()
    strong_negative = rf.compute_reward(strong_negative_portfolio, [], 0.0)
    rf.reset()
    mild_negative = rf.compute_reward(mild_negative_portfolio, [], 0.0)
    assert strong_profit > 0.5
    assert mild_profit > 0.00
    assert strong_negative < -0.5
    assert mild_negative < -0.00
    assert mild_profit < strong_profit
    assert mild_negative > strong_negative


def test_sharpe_ratio(
    strong_profitable_portfolio, strong_profitable_data, strong_negative_data
):
    cfg = RewardConfig(
        type="sharpe_ratio", reward_scaling=1e5, kwargs={"risk_free_rate": 0.01}
    )
    rew = reward_factory_method(cfg, initial_state=np.array([5000.0, 0.0]))

    strong_negative_data = strong_negative_data.droplevel("symbol")
    strong_profitable_data = strong_profitable_data.droplevel("symbol")

    r = rew.compute_reward(
        pf=strong_profitable_portfolio, df=strong_profitable_data, realized_profit=0.0
    )
    assert r > 1.0
    rew.reset()

    r = rew.compute_reward(
        pf=strong_negative_data, df=strong_negative_data, realized_profit=0.0
    )
    assert r < 0.0


def test_calmar_ratio(
    strong_profitable_portfolio, strong_profitable_data, strong_negative_data
):
    cfg = RewardConfig(
        type="calmar_ratio", reward_scaling=1e5, kwargs={"risk_free_rate": 0.01}
    )
    rew = reward_factory_method(cfg, initial_state=np.array([5000.0, 0.0]))

    strong_negative_data = strong_negative_data.droplevel("symbol")
    strong_profitable_data = strong_profitable_data.droplevel("symbol")

    r = rew.compute_reward(
        pf=strong_profitable_portfolio, df=strong_profitable_data, realized_profit=0.0
    )
    assert r > 1.0
    rew.reset()

    r = rew.compute_reward(
        pf=strong_negative_data, df=strong_negative_data, realized_profit=0.0
    )
    assert r < 0.0


def test_sortino_ratio(
    strong_profitable_portfolio, strong_profitable_data, strong_negative_data
):
    cfg = RewardConfig(
        type="sortino_ratio", reward_scaling=1e5, kwargs={"risk_free_rate": 0.01}
    )
    rew = reward_factory_method(cfg, initial_state=np.array([5000.0, 0.0]))

    strong_negative_data = strong_negative_data.droplevel("symbol")
    strong_profitable_data = strong_profitable_data.droplevel("symbol")

    r = rew.compute_reward(
        pf=strong_profitable_portfolio, df=strong_profitable_data, realized_profit=0.0
    )
    assert r > 1.0
    rew.reset()

    r = rew.compute_reward(
        pf=strong_negative_data, df=strong_negative_data, realized_profit=0.0
    )
    assert r < 0.0


def test_fast_profit_reward(
    strong_profitable_portfolio,
    mild_profitable_portfolio,
    strong_negative_portfolio,
    mild_negative_portfolio,
):
    cfg = RewardConfig(type="fast_profit_reward", reward_scaling=1000.0)
    rew = reward_factory_method(cfg, initial_state=np.array([5000.0, 0.0]))

    strong_profit = rew.compute_reward(strong_profitable_portfolio, pd.DataFrame(), 0.0)
    rew.reset()
    mild_profit = rew.compute_reward(mild_profitable_portfolio, pd.DataFrame(), 0.0)
    rew.reset()
    strong_negative = rew.compute_reward(strong_negative_portfolio, pd.DataFrame(), 0.0)
    rew.reset()
    mild_negative = rew.compute_reward(mild_negative_portfolio, pd.DataFrame(), 0.0)
    assert strong_profit > 0.5
    assert mild_profit > 0.00 and mild_profit < 0.5
    assert strong_negative < -0.5
    assert mild_negative < -0.00 and mild_negative > -0.5


def test_simple_momentum_reward(
    strong_profitable_portfolio,
    mild_profitable_portfolio,
    strong_negative_portfolio,
    mild_negative_portfolio,
):
    cfg = RewardConfig(type="simple_momentum_reward", reward_scaling=1000.0)
    rew = reward_factory_method(cfg, initial_state=np.array([5000.0, 0.0]))

    strong_profit = rew.compute_reward(strong_profitable_portfolio, pd.DataFrame(), 0.0)
    rew.reset()
    mild_profit = rew.compute_reward(mild_profitable_portfolio, pd.DataFrame(), 0.0)
    rew.reset()
    strong_negative = rew.compute_reward(strong_negative_portfolio, pd.DataFrame(), 0.0)
    rew.reset()
    mild_negative = rew.compute_reward(mild_negative_portfolio, pd.DataFrame(), 0.0)
    assert strong_profit > 0.5
    assert mild_profit > 0.00 and mild_profit < 0.5
    assert strong_negative < -0.5
    assert mild_negative < -0.00 and mild_negative > -0.5
