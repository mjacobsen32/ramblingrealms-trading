import pandas as pd
import vectorbt as vbt

from trading.cli.alg.config import RewardConfig
from trading.src.alg.portfolio.portfolio import Portfolio


class RewardFunction:
    """
    Base class for reward functions in trading environments.
    @TODO pass in the profit calculated from the position step
    """

    def __init__(self, cfg: RewardConfig, initial_state: pd.DataFrame):
        self.cfg = cfg
        self.initial_state = initial_state

    def __repr__(self) -> str:
        return f"RewardFunction(cfg={self.cfg}, initial_state={self.initial_state})"

    def reset(self):
        """
        Reset the reward function state.
        """
        pass

    def compute_reward(
        self, pf: Portfolio, df: pd.DataFrame, realized_profit: float
    ) -> float:
        """
        Compute the reward for a given action and state transition.
        """
        return 0.0
