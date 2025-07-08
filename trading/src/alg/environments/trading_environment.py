from typing import List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trading.cli.alg.config import StockEnv
from trading.src.features import utils as feature_utils
from trading.src.features.generic_features import Feature


class TradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, cfg: StockEnv, features: List[Feature]):
        self.unique_symbols = data["tic"].unique()
        self.stock_dimension = len(self.unique_symbols)
        self.feature_cols = feature_utils.get_feature_cols(features=features)
        self.data = data.copy()
        self.trade_limit_pct = 0.1  # max 10% of total assets per trade
        self.hmax = 10_000  # absolute max trade size in dollars

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
        self.max_steps = len(data["timestamp"].unique())
        # historical data
        self.asset_memory = [self.initial_cash]

        self._reset_internal_states()

    def _reset_internal_states(self):
        self.day = 0
        self.cash = self.initial_cash
        self.stock_owned = np.zeros(self.stock_dimension, dtype=np.float32)
        self.terminal = False
        self.total_assets = self.initial_cash

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_internal_states()
        return self._get_observation(), {}

    def _get_current_date(self):
        return self.data["timestamp"].unique()[self.day]

    def _get_day_prices(self):
        df_day = self.data[self.data["timestamp"] == self._get_current_date()]
        return df_day["close"].values

    def _get_observation(self):
        current_date = self._get_current_date()
        df_day = self.data[self.data["timestamp"] == current_date]
        df_day = df_day.set_index("tic").reindex(self.unique_symbols)

        prices = df_day["close"].fillna(0).values
        indicators = df_day[self.feature_cols].fillna(0).values.flatten()

        obs = np.concatenate([[self.cash], self.stock_owned, prices, indicators])
        return obs.astype(np.float32)

    def render(self):
        print(f"Day: {self.day}")
        print(f"Cash: {self.cash:.2f}")
        print(f"Stock Owned: {self.stock_owned}")
        print(f"Total Assets: {self.total_assets:.2f}")

    def step(self, actions):
        if self.terminal:
            return self._get_observation(), 0, self.terminal, False, {}

        current_date = self._get_current_date()
        df_day = self.data[self.data["timestamp"] == current_date]
        available_tics = df_day["tic"].values
        prices = df_day.set_index("tic")["close"].reindex(self.unique_symbols).values

        # Apply actions only to available tickers
        trade_limit = self.trade_limit_pct * self.total_assets
        scaled_actions = actions * trade_limit
        scaled_actions = np.clip(scaled_actions, -self.hmax, self.hmax)
        for i, symbol in enumerate(self.unique_symbols):
            price = prices[i]
            action = scaled_actions[i]

            if np.isnan(price):  # Ticker not available on this day
                continue

            if action < 0:  # Sell
                num_shares_to_sell = min(abs(action) // price, self.stock_owned[i])
                proceeds = num_shares_to_sell * price
                cost = proceeds * self.sell_cost[i]
                self.cash += proceeds - cost
                self.stock_owned[i] -= num_shares_to_sell

            elif action > 0:  # Buy
                num_shares_to_buy = action // price
                cost = num_shares_to_buy * price
                fee = cost * self.buy_cost[i]
                if cost + fee <= self.cash:
                    self.cash -= cost + fee
                    self.stock_owned[i] += num_shares_to_buy

        self.day += 1
        self.terminal = self.day >= self.max_steps - 1

        # Portfolio value: use available prices only
        valid_prices = np.nan_to_num(prices)
        portfolio_value = self.cash + np.sum(valid_prices * self.stock_owned)
        reward = portfolio_value - self.total_assets
        self.total_assets = portfolio_value
        self.asset_memory.append(self.total_assets)
        return self._get_observation(), reward, self.terminal, False, {}
