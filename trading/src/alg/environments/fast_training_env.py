import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit

from trading.cli.alg.config import StockEnv, TradeMode
from trading.src.alg.environments.base_environment import BaseTradingEnv
from trading.src.features.generic_features import Feature


class FastTrainingEnv(BaseTradingEnv):
    """
    Fast training environment with minimal state tracking.
    Optimized for speed with constant-time operations.
    Does NOT maintain position history, trade constraints, or complex metrics.
    Target: 10,000 iterations per second.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cfg: StockEnv,
        features: List[Feature],
        time_step: tuple[TimeFrameUnit, int] = (TimeFrameUnit.Day, 1),
    ):
        super().__init__(data, cfg, features, time_step)
        
        # Minimal state: just cash and current holdings (no history)
        self.initial_cash = cfg.portfolio_config.initial_cash
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.stock_dimension, dtype=np.float32)
        
        # Pre-compute price data for fast access
        self._precompute_price_arrays()
        
        logging.info("FastTrainingEnv initialized with %d symbols, lookback=%d",
                     self.stock_dimension, self.cfg.lookback_window)

    def _precompute_price_arrays(self):
        """Pre-compute price arrays and feature matrices for fast lookups."""
        # Create a 2D array: [timestep, symbol] for O(1) price lookups
        self.price_matrix = np.zeros((len(self.timestamps), self.stock_dimension), dtype=np.float32)
        
        # Pre-compute feature matrix: [timestep, features_per_symbol * num_symbols]
        # This avoids expensive dataframe operations during training
        num_features = len(self.feature_cols)
        self.feature_matrix = np.zeros(
            (len(self.timestamps), num_features * self.stock_dimension), 
            dtype=np.float32
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
        """
        if i == -1:
            i = self.observation_index
        
        # Get portfolio state (just cash and holdings, no complex calculations)
        portfolio_state = np.concatenate([[self.cash], self.holdings])
        
        # Get current prices from pre-computed matrix
        prices = self.price_matrix[i]
        
        # Get features for the lookback window from pre-computed matrix
        # Instead of slicing dataframe, we slice the feature matrix
        start_idx = max(0, i - self.cfg.lookback_window)
        end_idx = i + 1
        feature_window = self.feature_matrix[start_idx:end_idx].flatten()
        
        # Combine all observations
        observation = np.concatenate([portfolio_state, prices, feature_window], axis=0)
        
        logging.debug(
            "Portfolio state: %s\nPrices: %s\nFeatures: %s",
            portfolio_state.shape,
            prices.shape,
            feature_window.shape,
        )
        
        return observation.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset to initial state."""
        super().reset(seed=seed, options=options)
        self._reset_internal_states()
        self.cash = self.initial_cash
        self.holdings.fill(0.0)
        return self._get_observation(), {}

    def step(self, action):
        """
        Fast step with minimal state updates.
        Reward based on immediate portfolio value change.
        """
        if self.terminal:
            return (
                self._get_observation(self.observation_index - 1),
                0.0,
                self.terminal,
                False,
                {},
            )

        # Get current prices
        current_prices = self.price_matrix[self.observation_index]
        
        # Calculate portfolio value before action
        portfolio_value_before = self.cash + np.dot(self.holdings, current_prices)
        
        # Simple action scaling: convert normalized actions to share amounts
        # Action > threshold means buy, < -threshold means sell
        action_threshold = self.cfg.portfolio_config.action_threshold
        
        # Vectorized action processing
        buy_mask = action > action_threshold
        sell_mask = action < -action_threshold
        
        # Use configured trade limit percent for scaling
        trade_limit = self.cfg.portfolio_config.trade_limit_percent
        
        if self.cfg.portfolio_config.trade_mode == TradeMode.CONTINUOUS:
            # Continuous mode: scale actions by available capital/holdings
            max_buy_per_stock = (portfolio_value_before * trade_limit) / (current_prices + 1e-8)
            buy_amounts = np.where(buy_mask, action * max_buy_per_stock, 0.0)
            sell_amounts = np.where(sell_mask, action * self.holdings, 0.0)
        else:
            # Discrete mode: all-in or all-out
            max_shares = (portfolio_value_before * trade_limit) / (current_prices + 1e-8)
            buy_amounts = np.where(buy_mask, max_shares, 0.0)
            sell_amounts = np.where(sell_mask, -self.holdings, 0.0)
        
        # Combine buy and sell
        net_shares = buy_amounts + sell_amounts
        
        # Ensure we don't sell more than we have
        net_shares = np.clip(net_shares, -self.holdings, np.inf)
        
        # Update holdings and cash
        trade_cost = np.dot(net_shares, current_prices)
        
        # Ensure we have enough cash for buys
        if trade_cost > self.cash:
            # Scale down buys proportionally
            scale_factor = self.cash / (trade_cost + 1e-8)
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
        pct_change = (portfolio_value_after - portfolio_value_before) / (portfolio_value_before + 1e-8)
        
        # Normalize reward with tanh - use a default scaling factor for percentage changes
        # This makes the reward more sensitive to small changes (1% = 100 in scaling)
        reward = np.tanh(pct_change * 100.0)
        
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
