import logging
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit
from gymnasium import spaces

from trading.cli.alg.config import StockEnv, TradeMode
from trading.src.alg.environments.reward_functions.reward_function_factory import (
    reward_factory_method,
)
from trading.src.features import utils as feature_utils
from trading.src.features.generic_features import Feature
from trading.src.portfolio.portfolio import Portfolio, PositionManager

# Import the new environment classes
from trading.src.alg.environments.stateful_trading_env import StatefulTradingEnv


class TradingEnv(StatefulTradingEnv):
    """
    Trading environment for reinforcement learning agents.
    
    This class now inherits from StatefulTradingEnv and provides backward compatibility.
    For new code, consider using:
    - FastTrainingEnv for fast training with minimal state
    - StatefulTradingEnv for backtesting and paper trading with full state
    """
    pass
