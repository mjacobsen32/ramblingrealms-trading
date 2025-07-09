from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trading.cli.alg.config import StockEnv
from trading.src.features import utils as feature_utils
from trading.src.features.generic_features import Feature


class TradingEnv(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        cfg: StockEnv,
        features: List[Feature],
        backtest: bool = False,
    ):
        self.unique_symbols = data["tic"].unique()
        self.stock_dimension = len(self.unique_symbols)
        self.feature_cols = feature_utils.get_feature_cols(features=features)
        self.data = data.copy()
        self.trade_limit_pct = 0.1  # max 10% of total assets per trade
        self.hmax = 10_000  # absolute max trade size in dollars
        self.backtest = backtest

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
        self.position_queues: Dict = {
            symbol: [] for symbol in self.data["tic"].unique()
        }
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

    def calculate_normalized_profit(self, shares, price, tic):
        """
        Calculate normalized profit for a given trade.
        """
        if shares == 0:
            return 0.0
        # Calculate profit by matching shares from the position queue (FIFO)
        queue = self.position_queues[tic]
        shares_to_match = shares
        total_cost = 0.0
        total_initial_value = 0.0
        for idx, (s, p, f) in enumerate(queue):
            if shares_to_match <= 0:
                break
            matched_shares = min(s, shares_to_match)
            total_cost += matched_shares * price
            total_initial_value += matched_shares * p
            shares_to_match -= matched_shares
        if total_initial_value == 0 or shares_to_match > 0:
            return 0.0
        profit = total_cost - total_initial_value
        return profit / total_initial_value if total_initial_value != 0 else 0.0

    def reward_function(self):
        """
        Calculate the reward based on the current portfolio value.
        """
        current_date = self._get_current_date()
        df_day = self.data[self.data["timestamp"] == current_date]
        prices = df_day.set_index("tic")["close"].reindex(self.unique_symbols).values

        # Calculate portfolio value
        valid_prices = np.nan_to_num(prices)
        portfolio_value = self.cash + np.sum(valid_prices * self.stock_owned)

        # Calculate normalized profit for each stock
        normalized_profits = [
            self.calculate_normalized_profit(shares, price, symbol)
            for shares, price, symbol in zip(
                self.stock_owned, valid_prices, self.unique_symbols
            )
        ]

        # Reward is the change in portfolio value from the last step
        if len(self.asset_memory) > 1:
            previous_value = self.asset_memory[-2]
            reward = (
                (portfolio_value - previous_value) / previous_value
                if previous_value != 0
                else 0.0
            )
        else:
            reward = 0.0

        # Update asset memory
        self.asset_memory.append(portfolio_value)

        return reward + np.mean(normalized_profits)

    def step(self, actions):
        if self.terminal:
            return self._get_observation(), 0, self.terminal, False, {}

        current_date = self._get_current_date()
        df_day = self.data[self.data["timestamp"] == current_date]
        prices = df_day.set_index("tic")["close"].reindex(self.unique_symbols).values

        # Apply actions only to available tickers
        trade_limit = self.trade_limit_pct * self.total_assets
        scaled_actions = actions * trade_limit
        scaled_actions = np.clip(scaled_actions, -self.hmax, self.hmax)
        for i, symbol in enumerate(self.unique_symbols):
            price = prices[i]
            action = scaled_actions[i]
            dt_shares = 0

            if np.isnan(price):  # Ticker not available on this day
                continue

            if action < 0:  # Sell
                dt_shares = min(abs(action) // price, self.stock_owned[i])
                proceeds = dt_shares * price
                cost = proceeds * self.sell_cost[i]
                self.cash += proceeds - cost
                self.stock_owned[i] -= dt_shares
                dt_shares = -dt_shares  # Negative for selling

            elif action > 0:  # Buy
                dt_shares = action // price
                cost = dt_shares * price
                fee = cost * self.buy_cost[i]
                self.position_queues[symbol].append((dt_shares, price, fee))
                if cost + fee <= self.cash:
                    self.cash -= cost + fee
                    self.stock_owned[i] += dt_shares

            self.data.loc[
                (self.data["timestamp"] == current_date) & (self.data["tic"] == symbol),
                "size",
            ] = dt_shares

        self.day += 1
        self.terminal = self.day >= self.max_steps - 1

        # Portfolio value: use available prices only
        valid_prices = np.nan_to_num(prices)
        portfolio_value = self.cash + np.sum(valid_prices * self.stock_owned)

        self.total_assets = portfolio_value

        return self._get_observation(), self.reward_function(), self.terminal, False, {}
