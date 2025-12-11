from pathlib import Path

import numpy as np
import pytest

import trading.src.alg.agents
from trading.cli.alg.alg import backtest
from trading.cli.alg.config import RRConfig
from trading.src.alg.data_process.data_loader import DataLoader
from trading.src.alg.environments.trading_environment import TradingEnv
from trading.src.alg.environments.fast_training_env import FastTrainingEnv
from trading.src.alg.environments.stateful_trading_env import StatefulTradingEnv
from trading.test.alg.test_fixtures import *

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


def test_fast_training_env():
    """Test the fast training environment for speed optimization."""
    with Path.open(Path(CONFIG_DIR / "generic_alg.json")) as f:
        alg_config = RRConfig.model_validate_json(f.read())
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    
    # Create fast training environment
    fast_env = FastTrainingEnv(
        data=data_loader.df,
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    obs, info = fast_env.reset()
    
    # Verify observation shape
    assert obs.shape == fast_env.observation_space.shape
    
    # Test buy action
    action = np.ones(fast_env.stock_dimension) * 0.5
    obs, reward, terminated, truncated, info = fast_env.step(action)
    
    # Verify we have holdings now
    assert np.sum(fast_env.holdings) > 0
    assert "net_value" in info
    
    # Test hold
    action = np.zeros(fast_env.stock_dimension)
    obs, reward, terminated, truncated, info = fast_env.step(action)
    
    # Test sell
    action = np.ones(fast_env.stock_dimension) * -0.5
    obs, reward, terminated, truncated, info = fast_env.step(action)
    
    # Run through several steps
    for _ in range(10):
        action = np.random.uniform(-1, 1, fast_env.stock_dimension)
        obs, reward, terminated, truncated, info = fast_env.step(action)
        if terminated:
            break
    
    # Verify environment can reset
    obs, info = fast_env.reset()
    assert np.all(fast_env.holdings == 0)
    assert fast_env.cash == fast_env.initial_cash


def test_stateful_trading_env():
    """Test the stateful trading environment for accurate backtesting."""
    with Path.open(Path(CONFIG_DIR / "generic_alg.json")) as f:
        alg_config = RRConfig.model_validate_json(f.read())
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    
    # Create stateful trading environment
    stateful_env = StatefulTradingEnv(
        data=data_loader.df,
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    obs, info = stateful_env.reset()
    
    # Verify observation shape
    assert obs.shape == stateful_env.observation_space.shape
    
    # Verify portfolio is initialized
    assert stateful_env.pf is not None
    assert stateful_env.pf.position_manager is not None
    
    # Test buy action
    action = np.ones(stateful_env.stock_dimension) * 0.5
    obs, reward, terminated, truncated, info = stateful_env.step(action)
    
    # Verify portfolio state tracking
    assert "net_value" in info
    assert "profit_change" in info
    
    # Test sell
    action = np.ones(stateful_env.stock_dimension) * -0.5
    obs, reward, terminated, truncated, info = stateful_env.step(action)
    
    # Run through several steps
    for _ in range(10):
        action = np.random.uniform(-1, 1, stateful_env.stock_dimension)
        obs, reward, terminated, truncated, info = stateful_env.step(action)
        if terminated:
            break
    
    # Verify statistics are being tracked
    assert len(stateful_env.stats) > 0


def test_backward_compatibility():
    """Test that TradingEnv still works (backward compatibility)."""
    with Path.open(Path(CONFIG_DIR / "generic_alg.json")) as f:
        alg_config = RRConfig.model_validate_json(f.read())
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    
    # TradingEnv should still work and behave like StatefulTradingEnv
    env = TradingEnv(
        data=data_loader.df,
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    obs, info = env.reset()
    
    # Should have full portfolio tracking
    assert hasattr(env, "pf")
    assert hasattr(env, "stats")
    
    # Should be able to take steps
    action = np.ones(env.stock_dimension) * 0.5
    obs, reward, terminated, truncated, info = env.step(action)
    assert "net_value" in info
