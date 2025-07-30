import logging
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import vectorbt as vbt
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize

from trading.cli.alg.config import StockEnv
from trading.src.alg.environments.reward_functions.reward_function_factory import (
    factory_method,
)
from trading.src.alg.portfolio.portfolio import Portfolio
from trading.src.features import utils as feature_utils
from trading.src.features.generic_features import Feature


class TradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning agents.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cfg: StockEnv,
        features: List[Feature],
    ):
        self.stock_dimension = len(data.index.get_level_values("symbol").unique())
        self.feature_cols = feature_utils.get_feature_cols(features=features)
        self.init_data(data)
        self.cfg = cfg
        self.reward_function = factory_method(cfg.reward_config)
        self.pf: Portfolio = Portfolio(
            initial_cash=cfg.initial_cash,
            stock_dimension=self.stock_dimension,
        )
        self.observation_index = 0

        # Define action space: continuous [-1,1] per stock (sell, hold, buy)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dimension,), dtype=np.float32
        )

        # Environment state space
        state_space = (
            1  # cash balance
            + 2 * self.stock_dimension  # stock prices and shares held
            + (
                len(self.feature_cols) * self.stock_dimension
            )  # technical indicators for each stock
        )
        logging.info(
            f"State space: {state_space}, Features ({len(self.feature_cols)}): {self.feature_cols}, "
            f"Stock Dimension: {self.stock_dimension}, action space: {self.action_space.shape}"
        )
        # State space (cash + owned shares + prices + indicators)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32
        )

        self._reset_internal_states()

    def init_data(self, data: pd.DataFrame):
        """
        Set the data for the environment.
        Args:
            data: DataFrame containing the trading data.
        """
        self.data = data.copy()
        self.data = self.data.reorder_levels(["timestamp", "symbol"])
        self.data["size"] = 0  # Initialize size column for trades
        self.timestamps = data.index.get_level_values("timestamp").unique().to_list()
        self.max_steps = len(self.timestamps) - 1

    def _reset_internal_states(self):
        self.observation_index = 0
        self.terminal = False
        self.pf.reset()
        self.observation_timestamp = self.data.index.get_level_values(
            "timestamp"
        ).unique()
        logging.debug(f"Environment reset:\n{self.render()}")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed, options=options)
        self._reset_internal_states()
        return self._get_observation(), {}

    def _get_observation(self, i: int = -1) -> np.ndarray:
        """Get the current observation from the environment.

        Returns:
            np.ndarray: The current observation. [cash, positions, prices, indicators]
        """
        if i == -1:
            i = self.observation_index
        indicators = (
            self.data.loc[[self.observation_timestamp[i]]][self.feature_cols]
            .to_numpy()
            .flatten()
        )
        prices = self.data.loc[[self.observation_timestamp[i]]]["close"].to_numpy()
        portfolio_state = np.asarray(self.pf.state(self.observation_timestamp[i]))
        logging.debug(
            f"Portfolio state: {portfolio_state.shape}\nPrices: {prices.shape}\nIndicators: {indicators.shape}"
        )
        c = np.concatenate(
            [
                portfolio_state,
                prices,
                indicators,
            ]
        )

        logging.debug(f"Observation: {c}")

        return c

    def render(self):
        return (
            f"Day: {self.observation_timestamp[self.observation_index]}\n"
            f"Slice: {self.data.loc[self.observation_timestamp[self.observation_index]]}\n"
            f"Tickers: {self.data.index.get_level_values('symbol').unique().tolist()}, "
            f"Observation Space: {self.observation_space.shape}, "
            f"Action Space: {self.action_space.shape}, "
            f"Features: {self.feature_cols}, "
            f"Reward Function: {self.reward_function}, "
            f"{self.pf}"
        )

    def get_scaled_actions(self, action: np.ndarray) -> np.ndarray:
        attempted_buy = (
            action[action > 0]
            * self.data.loc[self.observation_timestamp[self.observation_index]][
                "close"
            ][action > 0]
        ).sum()
        if attempted_buy > self.pf.cash and np.any(action > 0):
            trade_limit = self.pf.cash / (action > 0).sum()
        else:
            trade_limit = self.cfg.trade_limit_percent * self.pf.total_value
        scaled_actions = (
            np.clip(action * trade_limit, -self.cfg.hmax, self.cfg.hmax)
            // self.data.loc[self.observation_timestamp[self.observation_index]][
                "close"
            ]
        )
        scaled_actions = np.nan_to_num(scaled_actions)
        logging.debug(f"Scaled Actions: {scaled_actions}")
        return scaled_actions

    def step(self, action):
        """
        Execute one time step within the environment.
        Args:
            action: The action to be taken by the agent, which is a vector of size equal to the number of stocks.
        Returns:
            observation: The new state of the environment after taking the action.
            reward: The reward received after taking the action.
            done: A boolean indicating whether the episode has ended.
            truncated: A boolean indicating whether the episode was truncated.
            info: Additional information about the environment.
        """

        if self.terminal:
            return (
                self._get_observation(self.observation_index - 1),
                0,
                self.terminal,
                False,
                {},
            )

        scaled_actions = self.get_scaled_actions(action)

        self.data.loc[self.observation_timestamp[self.observation_index], "size"] = (
            scaled_actions
            if isinstance(scaled_actions, np.ndarray)
            else scaled_actions.to_numpy()
        )
        logging.debug(
            f"{self.data.loc[self.observation_timestamp[self.observation_index], 'size']}"
        )
        self.pf.update_position_batch(
            self.data.loc[[self.observation_timestamp[self.observation_index]]]
        )

        ret_info = {"net_value": self.pf.net_value()}

        logging.debug(f"Env State: {self.render()}")
        logging.debug(f"Portfolio State: {self.pf}")
        ret = (
            self._get_observation(),
            self.reward_function.compute_reward(self.pf),
            self.terminal,
            False,
            ret_info,
        )
        self.observation_index += 1
        self.terminal = self.observation_index >= self.max_steps - 1
        return ret
