"""
Fast reward function optimized for training speed.
Does not maintain any history or complex calculations.
"""
import logging

import numpy as np
import pandas as pd

from trading.cli.alg.config import RewardConfig
from trading.src.alg.environments.reward_functions.base_reward_function import (
    RewardFunction,
)
from trading.src.portfolio.portfolio import Portfolio


class FastProfitReward(RewardFunction):
    """
    Ultra-fast reward function that only considers immediate portfolio value change.
    No history tracking, no complex metrics. Perfect for fast training.
    """

    def __init__(self, cfg: RewardConfig, initial_state: np.ndarray):
        super().__init__(cfg, initial_state=initial_state)
        self.previous_value = initial_state[0] if len(initial_state) > 0 else 100000.0

    def __repr__(self) -> str:
        return f"FastProfitReward(previous_value={self.previous_value})"

    def reset(self):
        """Reset reward state."""
        self.previous_value = (
            self.initial_state[0] if len(self.initial_state) > 0 else 100000.0
        )
        return super().reset()

    def compute_reward(
        self, pf: Portfolio, df: pd.DataFrame, realized_profit: float
    ) -> float:
        """
        Compute reward based on simple portfolio value change.
        
        Args:
            pf: Portfolio object (we extract net_value)
            df: DataFrame (not used in fast mode)
            realized_profit: Realized profit from trades (not used in fast mode)
            
        Returns:
            Normalized reward value
        """
        current_value = pf.position_manager.net_value()
        delta_value = current_value - self.previous_value
        
        # Update for next call
        self.previous_value = current_value
        
        # Normalize with tanh for stability
        normalized_reward = np.tanh(delta_value / self.cfg.reward_scaling)
        
        if np.isnan(normalized_reward) or np.isinf(normalized_reward):
            logging.warning("Invalid reward value: %s", normalized_reward)
            return 0.0
        
        return float(normalized_reward)


class SimpleMomentumReward(RewardFunction):
    """
    Reward based on price momentum (change in portfolio value relative to price movement).
    Still fast but slightly more sophisticated than pure profit.
    """

    def __init__(self, cfg: RewardConfig, initial_state: np.ndarray):
        super().__init__(cfg, initial_state=initial_state)
        self.previous_value = initial_state[0] if len(initial_state) > 0 else 100000.0

    def __repr__(self) -> str:
        return f"SimpleMomentumReward(previous_value={self.previous_value})"

    def reset(self):
        """Reset reward state."""
        self.previous_value = (
            self.initial_state[0] if len(self.initial_state) > 0 else 100000.0
        )
        return super().reset()

    def compute_reward(
        self, pf: Portfolio, df: pd.DataFrame, realized_profit: float
    ) -> float:
        """
        Compute reward that emphasizes profit relative to portfolio size.
        Encourages growing the portfolio while being mindful of risk.
        
        Args:
            pf: Portfolio object
            df: DataFrame (not used)
            realized_profit: Realized profit from trades
            
        Returns:
            Normalized reward value
        """
        current_value = pf.position_manager.net_value()
        
        # Calculate percentage return
        if self.previous_value > 0:
            pct_return = (current_value - self.previous_value) / self.previous_value
        else:
            pct_return = 0.0
        
        # Update for next call
        self.previous_value = current_value
        
        # Scale and normalize
        # Using a higher scaling makes the agent more sensitive to changes
        reward = np.tanh(pct_return * 100.0)
        
        if np.isnan(reward) or np.isinf(reward):
            logging.warning("Invalid reward value: %s", reward)
            return 0.0
        
        return float(reward)
