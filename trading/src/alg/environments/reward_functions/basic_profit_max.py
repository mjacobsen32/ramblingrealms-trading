import logging

import numpy as np

from trading.cli.alg.config import RewardConfig
from trading.src.alg.environments.reward_functions.base_reward_function import (
    RewardFunction,
)
from trading.src.alg.portfolio.portfolio import Portfolio


class BasicProfitMax(RewardFunction):
    def __init__(self, cfg: RewardConfig):
        self.previous_profit = 0.0
        super().__init__(cfg)

    def reset(self):
        self.previous_profit = 0.0
        return super().reset()

    def compute_reward(self, pf: Portfolio, current_date: str | None = None) -> float:
        if current_date is None:
            current_date = pf.df.index.get_level_values("timestamp")[-1]

        profit = (pf.net_value(current_date) - pf.initial_cash) / pf.initial_cash
        normalized_profit = np.tanh(profit)
        if (
            np.isnan(normalized_profit)
            or normalized_profit <= -1.0
            or normalized_profit >= 1.0
        ):
            logging.warning(f"Normalized profit is {normalized_profit}")
        ret = normalized_profit - self.previous_profit
        self.previous_profit = normalized_profit
        return ret
