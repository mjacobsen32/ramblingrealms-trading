import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit

from trading.cli.alg.config import StockEnv, TradeMode
from trading.src.alg.environments.base_environment import BaseTradingEnv
from trading.src.features.generic_features import Feature

# Constants
EPSILON = 1e-8  # Small value to prevent division by zero
PCT_TO_REWARD_SCALE = 100.0  # Scale factor: 1% change = 100, maps well to tanh(-1, 1)


class FastTrainingEnv(BaseTradingEnv):
    """
    Fast training environment with minimal state tracking.
    Optimized for speed with constant-time operations.
    Does NOT maintain position history, trade constraints, or complex metrics.
    Target: 10,000 iterations per second.

    Anti-memorization features:
    - Symbol shuffling at each reset to prevent learning position-specific patterns
    - hmax constraint to limit concentration in any single stock
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cfg: StockEnv,
        features: list[str] | list[Feature],
        time_step: tuple[TimeFrameUnit, int] = (TimeFrameUnit.Day, 1),
    ):
        super().__init__(data, cfg, features, time_step)

        # Minimal state: just cash and current holdings (no history)
        self.initial_cash = cfg.portfolio_config.initial_cash
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.stock_dimension, dtype=np.float32)

        # Symbol shuffling: maps action index -> actual symbol index in data
        # This prevents the model from memorizing "action[i] = best stock"
        self._symbol_permutation = np.arange(self.stock_dimension, dtype=np.int32)
        self._inverse_permutation = np.arange(self.stock_dimension, dtype=np.int32)

        # Pre-compute price data for fast access
        self._precompute_price_arrays()

        # Pre-compute hmax as shares limit per stock (for speed)
        self._hmax = cfg.portfolio_config.hmax

        logging.info(
            "FastTrainingEnv initialized with %d symbols, lookback=%d, hmax=%.2f",
            self.stock_dimension,
            self.cfg.lookback_window,
            self._hmax,
        )

    def _precompute_price_arrays(self):
        """Pre-compute price arrays and feature matrices for fast lookups."""
        # Create a 2D array: [timestep, symbol] for O(1) price lookups
        self.price_matrix = np.zeros(
            (len(self.timestamps), self.stock_dimension), dtype=np.float32
        )

        # Pre-compute feature matrix: [timestep, features_per_symbol * num_symbols]
        # This avoids expensive dataframe operations during training
        num_features = len(self.feature_cols)
        self.feature_matrix = np.zeros(
            (len(self.timestamps), num_features * self.stock_dimension),
            dtype=np.float32,
        )

        for i, ts in enumerate(self.timestamps):
            data_slice = self.data.loc[[ts]]
            prices = data_slice["price"].to_numpy()
            self.price_matrix[i, :] = prices

            # Flatten features for this timestep
            features = data_slice[self.feature_cols].to_numpy().flatten()
            self.feature_matrix[i, :] = features

    def _get_observation(self, i: int = -1) -> np.ndarray:
        """
        Get observation with minimal computation using pre-computed matrices.
        Returns: [cash, holdings, current_prices, indicators]

        Note: Holdings and prices are returned in SHUFFLED order matching the
        current symbol permutation, so the model sees a consistent view.
        """
        if i == -1:
            i = self.observation_index

        # Get portfolio state with shuffled holdings order
        # This ensures the model sees holdings in the same order as actions
        shuffled_holdings = self.holdings[self._symbol_permutation]
        portfolio_state = np.concatenate([[self.cash], shuffled_holdings])

        # Get current prices from pre-computed matrix, shuffled to match action order
        prices = self.price_matrix[i][self._symbol_permutation]

        # Get features for the lookback window from pre-computed matrix
        # Shuffle features to match the symbol permutation
        start_idx = max(0, i - self.cfg.lookback_window)
        end_idx = i + 1
        feature_window = self._get_shuffled_features(start_idx, end_idx)

        # Combine all observations
        observation = np.concatenate([portfolio_state, prices, feature_window], axis=0)

        logging.debug(
            "Portfolio state: %s\nPrices: %s\nFeatures: %s",
            portfolio_state.shape,
            prices.shape,
            feature_window.shape,
        )

        return observation.astype(np.float32)

    def _get_shuffled_features(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get features for the lookback window, shuffled to match symbol permutation.
        Maintains speed by using pre-computed indices.
        """
        num_features = len(self.feature_cols)
        raw_features = self.feature_matrix[
            start_idx:end_idx
        ]  # [timesteps, features*symbols]

        # Reshape to [timesteps, symbols, features], shuffle symbols, then flatten
        timesteps = raw_features.shape[0]
        reshaped = raw_features.reshape(timesteps, self.stock_dimension, num_features)
        shuffled = reshaped[:, self._symbol_permutation, :]
        return shuffled.flatten()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset to initial state with symbol shuffling.

        Symbol shuffling prevents the model from memorizing that a specific
        action index corresponds to the best-performing stock. Each episode,
        the mapping between action indices and actual stocks is randomized.
        """
        super().reset(seed=seed, options=options)
        self._reset_internal_states()
        self.cash = self.initial_cash
        self.holdings.fill(0.0)

        # Shuffle symbol order to prevent memorization
        # This randomizes which action index maps to which stock
        rng = np.random.default_rng(seed)
        self._symbol_permutation = rng.permutation(self.stock_dimension).astype(
            np.int32
        )
        # Compute inverse permutation for mapping actions back to data indices
        self._inverse_permutation = np.argsort(self._symbol_permutation).astype(
            np.int32
        )

        logging.debug("Reset with symbol permutation: %s", self._symbol_permutation[:5])

        return self._get_observation(), {}

    def step(self, action):
        """
        Fast step with minimal state updates.
        Reward based on immediate portfolio value change.

        Actions are mapped through the symbol permutation to actual stock indices.
        hmax constraint limits maximum shares traded per stock per step.
        """
        if self.terminal:
            return (
                self._get_observation(self.observation_index - 1),
                0.0,
                self.terminal,
                False,
                {},
            )

        # Get current prices (in original data order, not shuffled)
        current_prices = self.price_matrix[self.observation_index]

        # Calculate portfolio value before action
        portfolio_value_before = self.cash + np.dot(self.holdings, current_prices)

        # Map actions from shuffled order back to original data order
        # action[i] corresponds to symbol at _symbol_permutation[i]
        # We need to reorder actions to match the original data order
        action_in_data_order = action[self._inverse_permutation]

        # Simple action scaling: convert normalized actions to share amounts
        # Action > threshold means buy, < -threshold means sell
        action_threshold = self.cfg.portfolio_config.action_threshold

        # Vectorized action processing (now in data order)
        buy_mask = action_in_data_order > action_threshold
        sell_mask = action_in_data_order < -action_threshold

        # Use configured trade limit percent for scaling
        trade_limit = self.cfg.portfolio_config.trade_limit_percent

        if self.cfg.portfolio_config.trade_mode == TradeMode.CONTINUOUS:
            # Continuous mode: scale actions by available capital/holdings
            max_buy_per_stock = (portfolio_value_before * trade_limit) / (
                current_prices + EPSILON
            )
            buy_amounts = np.where(
                buy_mask, action_in_data_order * max_buy_per_stock, 0.0
            )
            sell_amounts = np.where(
                sell_mask, action_in_data_order * self.holdings, 0.0
            )
        else:
            # Discrete mode: all-in or all-out
            max_shares = (portfolio_value_before * trade_limit) / (
                current_prices + EPSILON
            )
            buy_amounts = np.where(buy_mask, max_shares, 0.0)
            sell_amounts = np.where(sell_mask, -self.holdings, 0.0)

        # Combine buy and sell
        net_shares = buy_amounts + sell_amounts

        # Apply hmax constraint: limit maximum shares traded per stock per step
        # This prevents over-concentration in any single stock
        hmax_in_shares = self._hmax / (current_prices + EPSILON)
        net_shares = np.clip(net_shares, -hmax_in_shares, hmax_in_shares)

        # Ensure we don't sell more than we have
        net_shares = np.clip(net_shares, -self.holdings, np.inf)

        # Update holdings and cash
        trade_cost = np.dot(net_shares, current_prices)

        # Ensure we have enough cash for buys
        if trade_cost > self.cash:
            # Scale down buys proportionally
            scale_factor = self.cash / (trade_cost + EPSILON)
            net_shares = np.where(net_shares > 0, net_shares * scale_factor, net_shares)
            trade_cost = np.dot(net_shares, current_prices)

        self.holdings += net_shares
        self.cash -= trade_cost

        # Move to next timestep
        self.observation_index += 1
        self.terminal = self.observation_index >= self.max_steps - 1

        # Calculate portfolio value after action
        if not self.terminal:
            next_prices = self.price_matrix[self.observation_index]
        else:
            next_prices = current_prices

        portfolio_value_after = self.cash + np.dot(self.holdings, next_prices)

        # Simple reward: percentage change in portfolio value
        pct_change = (portfolio_value_after - portfolio_value_before) / (
            portfolio_value_before + EPSILON
        )

        # Normalize reward with tanh - scale percentage changes to map well to tanh range
        # This makes the reward sensitive to small changes (1% = 100 in scaling)
        reward = np.tanh(pct_change * PCT_TO_REWARD_SCALE)

        info = {
            "net_value": portfolio_value_after,
            "cash": self.cash,
        }

        return (
            self._get_observation(),
            float(reward),
            self.terminal,
            False,
            info,
        )
