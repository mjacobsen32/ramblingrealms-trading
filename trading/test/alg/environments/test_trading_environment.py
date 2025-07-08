from pathlib import Path

import pytest

import trading.src.alg.agents
from trading.cli.alg.config import AlgConfig
from trading.src.alg.data_process.data_loader import DataLoader
from trading.src.alg.environments.trading_environment import TradingEnv


def test_env():
    with Path.open(
        Path(
            "/home/matthew-jacobsen/dev/ramblingrealms-trading/trading/configs/generic_alg.json"
        )
    ) as f:
        alg_config = AlgConfig.model_validate_json(f.read())
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    train_env = TradingEnv(
        data=data_loader.df,
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    train_env.reset()
    train_env.render()
    train_env.step([1 for _ in range(train_env.stock_dimension)])
    train_env.render()
