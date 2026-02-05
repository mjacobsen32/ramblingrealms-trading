import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit
from gymnasium import spaces

from trading.cli.alg.config import StockEnv, TradeMode
from trading.src.features import utils as feature_utils
from trading.src.features.generic_features import Feature


class BaseTradingEnv(gym.Env, ABC):
    """
    Base trading environment for reinforcement learning agents.
    Contains common functionality shared between training and testing environments.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cfg: StockEnv,
        features: list[Feature] | list[str],
        time_step: tuple[TimeFrameUnit, int] = (TimeFrameUnit.Day, 1),
    ):
        super().__init__()
        self.symbols = data.index.get_level_values("symbol").unique().tolist()
        self.stock_dimension = data.index.get_level_values("symbol").nunique()
        if len(self.symbols) != self.stock_dimension:
            raise ValueError("Data symbols length does not match stock dimension.")
        self.feature_cols = feature_utils.get_feature_cols(features)
        self.init_data(data)
        self.cfg: StockEnv = cfg
        self.time_step = time_step

        # Define action space: continuous [-1,1] per stock (sell, hold, buy)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.stock_dimension,),
            dtype=(
                np.float32
                if self.cfg.portfolio_config.trade_mode == TradeMode.CONTINUOUS
                else np.int32
            ),
        )

        # Environment state space
        state_space = (
            1  # cash balance
            + 2 * self.stock_dimension  # stock prices and shares held
            + (
                (len(self.feature_cols) * self.stock_dimension)
                * (self.cfg.lookback_window + 1)
            )  # technical indicators for each stock
        )
        logging.info(
            "State space: %s, Features (%s): %s, Stock Dimension: %s, action space: %s, lookback window: %i",
            state_space,
            len(self.feature_cols),
            self.feature_cols,
            self.stock_dimension,
            self.action_space.shape,
            self.cfg.lookback_window,
        )
        # State space (cash + owned shares + prices + indicators)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32
        )

        self.observation_index = self.cfg.lookback_window
        self.terminal = False
        # Initialize observation_timestamp early to avoid AttributeError
        self.observation_timestamp = None

    def init_data(self, data: pd.DataFrame):
        """
        Set the data for the environment.
        Args:
            data: DataFrame containing the trading data.
        """
        self.data = data.copy()
        self.data["timestamp"] = self.data.index.get_level_values("timestamp")
        self.timestamps = data.index.get_level_values("timestamp").unique().to_list()
        self.max_steps = len(self.timestamps)
        # Initialize observation_timestamp here so it's available immediately
        self.observation_timestamp = self.data.index.get_level_values(
            "timestamp"
        ).unique()

    @classmethod
    def observation(
        cls,
        df: pd.DataFrame,
        portfolio_state: np.ndarray,
        feature_cols: list,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Build observation from dataframe, portfolio state, and prices.
        """
        indicators = df[feature_cols].to_numpy().flatten()
        logging.debug(
            "Portfolio state: %s\nPrices: %s\nIndicators: %s",
            portfolio_state.shape,
            prices.shape,
            indicators.shape,
        )
        c = np.concatenate([portfolio_state, prices, indicators], axis=0)
        logging.debug("Observation: %s", c)
        return c

    def _get_observation_df(self, i: int = -1) -> pd.DataFrame:
        """Get the dataframe slice for observation at index i."""
        if i == -1:
            i = self.observation_index
        if self.observation_timestamp is None:
            raise ValueError("observation_timestamp not initialized")
        return self.data.loc[
            self.observation_timestamp[
                i - self.cfg.lookback_window
            ] : self.observation_timestamp[i]
        ]

    def _get_prices(self, i: int = -1) -> np.ndarray:
        """Get prices at index i."""
        if i == -1:
            i = self.observation_index
        if self.observation_timestamp is None:
            raise ValueError("observation_timestamp not initialized")
        return self.data.loc[[self.observation_timestamp[i]]]["price"].to_numpy()

    @abstractmethod
    def _get_observation(self, i: int = -1) -> np.ndarray:
        """Get the current observation from the environment."""
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to its initial state. Can be overridden by subclasses."""
        # Default implementation - subclasses should override
        pass

    @abstractmethod
    def step(self, action):
        """Execute one time step within the environment."""
        pass

    def render(self):
        """Render the environment state."""
        if self.observation_timestamp is None:
            return "Environment not initialized"
        return "Day: {}\nSlice: {}\nTickers: {}, Observation Space: {}, Action Space: {}, Features: {}".format(
            self.observation_timestamp[self.observation_index],
            self.data.loc[self.observation_timestamp[self.observation_index]],
            self.data.index.get_level_values("symbol").unique().tolist(),
            self.observation_space.shape,
            self.action_space.shape,
            self.feature_cols,
        )

    def _reset_internal_states(self, timestamp: pd.Timestamp | None = None):
        """Reset internal state counters."""
        self.terminal = False
        # Ensure observation_timestamp is always set
        if self.observation_timestamp is None or len(self.observation_timestamp) == 0:
            self.observation_timestamp = self.data.index.get_level_values(
                "timestamp"
            ).unique()
        if timestamp is not None:
            obs_ts = self.observation_timestamp
            if obs_ts is None:
                raise ValueError("observation_timestamp not initialized")
            obs_tz = getattr(obs_ts, "tz", None)
            if timestamp.tzinfo is None and obs_tz is not None:
                timestamp = timestamp.tz_localize(obs_tz)
            elif timestamp.tzinfo is not None and obs_tz is None:
                timestamp = timestamp.tz_convert(None)
            # pick the last timestamp <= requested timestamp
            ts_idx = obs_ts.searchsorted(timestamp, side="right") - 1
            if ts_idx < 0:
                raise ValueError(
                    f"Timestamp {timestamp} precedes available data starting at {obs_ts.min()}"
                )
            self.observation_index = ts_idx
        else:
            self.observation_index = self.cfg.lookback_window
        logging.debug("Environment reset:\n%s", self.render())
