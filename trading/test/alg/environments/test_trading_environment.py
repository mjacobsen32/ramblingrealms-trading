from pathlib import Path

import numpy as np
import pytest

import trading.src.alg.agents
from trading.cli.alg.alg import backtest
from trading.cli.alg.config import RRConfig
from trading.src.alg.data_process.data_loader import DataLoader
from trading.src.alg.environments.trading_environment import TradingEnv

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


def test_env():
    with Path.open(Path(CONFIG_DIR / "generic_alg.json")) as f:
        alg_config = RRConfig.model_validate_json(f.read())
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    train_env = TradingEnv(
        data=data_loader.df,
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    train_env.reset()
    # initial state
    train_env.render()

    # buy 10% of each
    train_env.step(np.ones(train_env.stock_dimension))
    train_env.render()

    # hold for a few weeks
    [
        train_env.step(np.array([0 for _ in range(train_env.stock_dimension)]))
        for _ in range(20)
    ]
    train_env.render()

    # sell 10% of each
    train_env.step(np.array([-1 for _ in range(train_env.stock_dimension)]))
    train_env.render()

    # attempt to sell more than we own
    train_env.step(np.array([-1 for _ in range(train_env.stock_dimension)]))
    train_env.render()
