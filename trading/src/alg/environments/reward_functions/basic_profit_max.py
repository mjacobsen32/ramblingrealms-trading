import logging

import numpy as np
import pandas as pd

from trading.cli.alg.config import RewardConfig
from trading.src.alg.environments.reward_functions.base_reward_function import (
    RewardFunction,
)
from trading.src.alg.portfolio.portfolio import Portfolio


class BasicProfitMax(RewardFunction):
    def __init__(self, cfg: RewardConfig, initial_state: np.ndarray):
        super().__init__(cfg, initial_state=initial_state)
        self.initial_net = initial_state[0]
        self.previous_net = self.initial_net

    def __repr__(self) -> str:
        return f"BasicProfitMax(profit_memory={self.previous_net})"

    def reset(self):
        self.previous_net = self.initial_net
        return super().reset()

    def compute_reward(
        self, pf: Portfolio, df: pd.DataFrame, realized_profit: float
    ) -> float:
        current_net = pf.net_value()
        delta_net = current_net - self.previous_net
        self.previous_net = current_net

        normalized_delta = np.tanh(delta_net / self.cfg.reward_scaling)
        if (
            np.isnan(normalized_delta)
            or normalized_delta <= -1.0
            or normalized_delta >= 1.0
        ):
            logging.warning("Normalized dt net value is %s", normalized_delta)
        return normalized_delta


class BasicRealizedProfitMax(RewardFunction):
    def __init__(self, cfg: RewardConfig, initial_state: np.ndarray):
        super().__init__(cfg, initial_state=initial_state)

    def compute_reward(
        self, pf: Portfolio, df: pd.DataFrame, realized_profit: float
    ) -> float:
        normalized_profit = np.tanh(realized_profit / self.cfg.reward_scaling)
        if (
            np.isnan(normalized_profit)
            or normalized_profit <= -1.0
            or normalized_profit >= 1.0
        ):
            logging.warning("Normalized profit is %s", normalized_profit)
        return normalized_profit
