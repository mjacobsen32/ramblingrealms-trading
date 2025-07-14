import vectorbt as vbt

from trading.cli.alg.config import RewardConfig
from trading.src.alg.portfolio.portfolio import Portfolio


class RewardFunction:
    """
    Base class for reward functions in trading environments.
    """

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def __repr__(self) -> str:
        return f"RewardFunction(cfg={self.cfg})"

    def reset(self):
        """
        Reset the reward function state.
        """
        pass

    def compute_reward(self, pf: Portfolio, current_date: str | None = None) -> float:
        """
        Compute the reward for a given action and state transition.
        """
        return 0.0
