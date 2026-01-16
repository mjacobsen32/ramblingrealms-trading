import pytest

from trading.cli.alg.config import TradeMode
from trading.cli.trading.trade_config import BrokerType
from trading.test.features.test_features import FillStrategy


@pytest.fixture(autouse=True)
def single_apple_ticker_request():
    from trading.cli.alg.config import DataRequests, DataSourceType

    return DataRequests(
        dataset_name="TEST_APPLE",
        source=DataSourceType.ALPACA,
        endpoint="StockBarRequest",
        kwargs={"symbol_or_symbols": ["AAPL"], "adjustment": "split"},
    )


@pytest.fixture
def data_config(single_apple_ticker_request):
    from alpaca.data.timeframe import TimeFrameUnit

    from trading.cli.alg.config import DataConfig

    return DataConfig(
        start_date="2023-01-01",
        end_date="2023-12-31",
        time_step_unit=TimeFrameUnit.Day,
        cache_path="{PROJECT_ROOT}/trading/test/data/",
        requests=[single_apple_ticker_request],
        validation_split=0.2,
        cache_enabled=True,
        time_step_period=1,
    )


@pytest.fixture
def feature_config():
    from alpaca.data.timeframe import TimeFrameUnit

    from trading.cli.alg.config import FeatureConfig
    from trading.src.features.generic_features import (
        Candle,
        FeatureType,
        FillStrategy,
    )

    c = Candle(
        type=FeatureType.CANDLE,
        name="candle",
        source="TEST_APPLE",
        fill_strategy=FillStrategy.DROP,
        enabled=True,
        period=TimeFrameUnit.Day,
    )
    return FeatureConfig(
        features=[c],
        fill_strategy="mean",
    )


@pytest.fixture
def data_loader(data_config, feature_config):
    from trading.src.alg.data_process.data_loader import DataLoader

    data = DataLoader(data_config=data_config, feature_config=feature_config)
    data.df["timestamp"] = data.df.index.get_level_values("timestamp")
    data.df["size"] = 0.0
    data.df["profit"] = 0.0
    data.df["action"] = 0.0
    return data


@pytest.fixture
def incomplete_multi_ticker_data_loader(data_config, feature_config):
    from trading.src.alg.data_process.data_loader import DataLoader

    data_config.requests[0].kwargs["symbol_or_symbols"] = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "HG",  # IPO'd on November 10th 2023
    ]
    data_config.requests[0].dataset_name = "TEST_INCOMPLETE_MULTI_TICKERS"
    feature_config.features[0].source = "TEST_INCOMPLETE_MULTI_TICKERS"
    feature_config.features[0].fill_strategy = FillStrategy.BACKWARD_FILL

    data = DataLoader(data_config=data_config, feature_config=feature_config)

    data.df["timestamp"] = data.df.index.get_level_values("timestamp")
    data.df["size"] = 0.0
    data.df["profit"] = 0.0
    data.df["action"] = 0.0
    return data


@pytest.fixture
def multi_data_loader(data_config, feature_config):
    from trading.src.alg.data_process.data_loader import DataLoader

    data_config.requests[0].kwargs["symbol_or_symbols"] = [
        "AAPL",
        "MSFT",
        "GOOGL",
    ]
    data_config.requests[0].dataset_name = "TEST_MULTI_TICKERS"
    feature_config.features[0].source = "TEST_MULTI_TICKERS"
    data = DataLoader(data_config=data_config, feature_config=feature_config)
    data.df["price"] = data.df["close"]
    data.df["timestamp"] = data.df.index.get_level_values("timestamp")
    data.df["size"] = 0.0
    data.df["profit"] = 0.0
    data.df["action"] = 0.0
    return data


@pytest.fixture
def agent_config():
    """
    Fixture to create a simple agent configuration.
    """
    from trading.cli.alg.config import AgentConfig

    return AgentConfig(
        algo="ppo",
        save_path="{PROJECT_ROOT}/trading/test/models/ppo.zip",
        deterministic=True,
        kwargs={"policy": "MlpPolicy", "n_steps": 2048},
        total_timesteps=1000,
    )


@pytest.fixture
def portfolio_config():
    """
    Fixture to create a simple portfolio configuration.
    """
    from trading.cli.alg.config import PortfolioConfig

    return PortfolioConfig(
        initial_cash=1_000_000,
        hmax=10_000,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        max_positions=None,
        trade_mode=TradeMode.CONTINUOUS,
        trade_limit_percent=0.1,
        action_threshold=0.01,
        maintain_history=True,
    )


@pytest.fixture
def reward_config():
    """
    Fixture to create a simple reward configuration.
    """
    from trading.cli.alg.config import RewardConfig

    return RewardConfig(type="basic_profit_max", reward_scaling=1e5)


@pytest.fixture
def stock_env_config(portfolio_config, reward_config):
    """
    Fixture to create a simple environment configuration.
    """
    from trading.cli.alg.config import StockEnv

    return StockEnv(
        portfolio_config=portfolio_config,
        reward_config=reward_config,
        turbulence_threshold=None,  # No turbulence threshold for testing
    )


@pytest.fixture
def trade_env(data_loader, stock_env_config, feature_config):
    from trading.src.alg.environments.trading_environment import TradingEnv

    return TradingEnv(
        data=data_loader.get_train_test()[0],
        cfg=stock_env_config,
        features=feature_config.features,
    )


@pytest.fixture
def agent(agent_config, trade_env):
    """
    Fixture to create a simple agent instance.
    """
    from trading.src.alg.agents.agents import Agent

    return Agent(agent_config, env=trade_env, load=False)


@pytest.fixture
def backtest(agent, data_loader, trade_env):
    """
    Fixture to create a backtest instance.
    """
    from trading.cli.alg.config import BackTestConfig
    from trading.src.alg.backtest.backtesting import BackTesting

    return BackTesting(
        model=agent.model,
        env=trade_env,
        backtest_config=BackTestConfig(),
        data=data_loader.get_train_test()[1],
    )


@pytest.fixture
def remote_trade_config(portfolio_config):
    """
    Fixture to create a remote trade configuration.
    """
    from trading.cli.trading.trade_config import RRTradeConfig

    return RRTradeConfig(
        model_path="{PROJECT_ROOT}/trading/test/models/ppo.zip",
        out_dir="{PROJECT_ROOT}/trading/test/temp/out/",
        positions_path="paper_trading/positions/ppo.json",
        account_path="paper_trading/accounts/ppo.json",
        closed_positions_path="paper_trading/closed_positions/ppo.parquet",
        account_value_series_path="paper_trading/account_value_series/ppo.parquet",
        broker=BrokerType.REMOTE,
        bucket_name="rr-storage",
        broker_kwargs={"service_name": "s3"},
        defer_trade_execution=False,
        portfolio_config=portfolio_config,
    )


@pytest.fixture
def local_trade_config(portfolio_config):
    """
    Fixture to create a remote trade configuration.
    """
    from trading.cli.trading.trade_config import RRTradeConfig

    return RRTradeConfig(
        model_path="{PROJECT_ROOT}/trading/test/models/ppo.zip",
        out_dir="{PROJECT_ROOT}/trading/test/temp/out/",
        positions_path="{PROJECT_ROOT}/trading/test/paper_trading/positions/ppo.json",
        account_path="{PROJECT_ROOT}/trading/test/paper_trading/accounts/ppo.json",
        closed_positions_path="{PROJECT_ROOT}/trading/test/paper_trading/closed_positions/ppo.parquet",
        account_value_series_path="{PROJECT_ROOT}/trading/test/paper_trading/account_value_series/ppo.parquet",
        meta_data_path="{PROJECT_ROOT}/trading/test/paper_trading/meta_data/ppo.json",
        broker=BrokerType.LOCAL,
        bucket_name=None,
        broker_kwargs={},
        defer_trade_execution=False,
        portfolio_config=portfolio_config,
    )


@pytest.fixture
def alpaca_trade_config(portfolio_config):
    """
    Fixture to create a remote trade configuration.
    """
    from trading.cli.trading.trade_config import RRTradeConfig

    return RRTradeConfig(
        model_path="{PROJECT_ROOT}/trading/test/models/ppo.zip",
        out_dir="{PROJECT_ROOT}/trading/test/temp/out/",
        broker=BrokerType.ALPACA,
        bucket_name=None,
        broker_kwargs={},
        defer_trade_execution=False,
        portfolio_config=portfolio_config,
    )


@pytest.fixture
def alpaca_trading_client_mock():
    from trading.test.mocks.alpaca_trading_client_mock import AlpacaTradingClientMock

    return AlpacaTradingClientMock()
