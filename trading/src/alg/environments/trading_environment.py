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
        backtest: bool = False,
    ):
        self.unique_symbols = data.index.get_level_values("symbol").unique().unique()
        self.stock_dimension = len(self.unique_symbols)
        self.feature_cols = feature_utils.get_feature_cols(features=features)
        self.data = data.copy()
        self.data = self.data.reorder_levels(["timestamp", "symbol"])
        self.data["size"] = 0  # Initialize size column for trades
        self.backtest = backtest
        self.cfg = cfg
        self.reward_function = factory_method(cfg.reward_config)
        self.pf: Portfolio = Portfolio(
            initial_cash=cfg.initial_cash,
        )

        # Environment state space
        self.state_space = (
            1  # cash balance
            + 2 * self.stock_dimension  # stock prices and shares held
            + len(self.feature_cols)
            * self.stock_dimension  # technical indicators for each stock
        )

        # Define action space: continuous [-1,1] per stock (sell, hold, buy)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dimension,), dtype=np.float32
        )

        # State space (cash + owned shares + prices + indicators)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )

        self.sell_cost = (
            [float(x) for x in cfg.sell_cost_pct]
            if isinstance(cfg.sell_cost_pct, list)
            else [float(cfg.sell_cost_pct)] * self.stock_dimension
        )
        self.buy_cost = (
            [float(x) for x in cfg.buy_cost_pct]
            if isinstance(cfg.buy_cost_pct, list)
            else [float(cfg.buy_cost_pct)] * self.stock_dimension
        )

        # Trading parameters
        self.initial_cash = cfg.initial_cash
        self.timestamps = data.index.get_level_values("timestamp").unique().to_list()
        # self.timestamps = pd.Index(self.data.index.get_level_values("timestamp"))
        self.max_steps = len(self.timestamps) - 1
        # historical data
        self.asset_memory = [self.initial_cash]
        self._reset_internal_states()

    def _reset_internal_states(self):
        self.day = 0
        self.cash = self.initial_cash
        self.stock_owned = np.zeros(self.stock_dimension, dtype=np.float32)
        self.terminal = False
        self.total_assets = self.initial_cash
        self.pf.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed, options=options)
        self._reset_internal_states()
        return self._get_observation(), {}

    def _get_current_date(self):
        return self.timestamps[self.day]

    def _get_day_prices(self):
        return self.data.xs(self._get_current_date(), level="timestamp")["close"].values

    def _get_observation(self):
        current_date = self._get_current_date()
        df_day = self.data.loc[[current_date]]
        prices = df_day["close"].fillna(0).values
        indicators = df_day[self.feature_cols].fillna(0).values.flatten()

        obs = np.concatenate([[self.cash], self.stock_owned, prices, indicators])
        return obs.astype(np.float32)

    def render(self):
        logging.info(f"Day: {self.day}")
        logging.info(f"Cash: {self.cash:.2f}")
        logging.info(f"Stock Owned: {self.stock_owned}")
        logging.info(f"Total Assets: {self.total_assets:.2f}")

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
            return self._get_observation(), 0, self.terminal, False, {}

        current_date = self._get_current_date()
        prices = self._get_day_prices()
        trade_limit = self.cfg.trade_limit_percent * self.pf.total_value
        scaled_actions = action * trade_limit
        scaled_actions = np.clip(scaled_actions, -self.cfg.hmax, self.cfg.hmax)
        for i, symbol in enumerate(self.unique_symbols):
            price = prices[i]
            act = scaled_actions[i]
            dt_shares = 0

            if np.isnan(price):  # Ticker not available on this day
                continue

            if act < 0:  # Sell
                dt_shares = min(abs(act) // price, self.stock_owned[i])
                proceeds = dt_shares * price
                cost = proceeds * self.sell_cost[i]
                self.cash += proceeds - cost
                self.stock_owned[i] -= dt_shares
                dt_shares = -dt_shares  # Negative for selling

            elif act > 0:  # Buy
                dt_shares = act // price
                cost = dt_shares * price
                fee = cost * self.buy_cost[i]
                if cost + fee <= self.cash:
                    self.cash -= cost + fee
                    self.stock_owned[i] += dt_shares

            self.data.loc[(current_date, symbol), "size"] = dt_shares
        self.day += 1
        self.terminal = self.day >= self.max_steps - 1

        self.pf.update_position_batch(self.data.loc[[current_date]])
        ret_info = {"net_value": self.pf.net_value(current_date)}
        return (
            self._get_observation(),
            self.reward_function.compute_reward(self.pf, current_date),
            self.terminal,
            False,
            ret_info,
        )
