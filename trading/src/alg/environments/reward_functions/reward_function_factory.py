import numpy as np

from trading.cli.alg.config import RewardConfig
from trading.src.alg.environments.reward_functions.base_reward_function import (
    RewardFunction,
)
from trading.src.alg.environments.reward_functions.basic_profit_max import (
    BasicProfitMax,
    BasicRealizedProfitMax,
    CalmarRatio,
    SharpeRatio,
    SortinoRatio,
)


def factory_method(cfg: RewardConfig, initial_state: np.ndarray) -> RewardFunction:
    """
    Factory method to create a reward function instance based on the configuration.
    """
    if cfg.type == "basic_profit_max":
        return BasicProfitMax(cfg, initial_state=initial_state)
    elif cfg.type == "realized_profit_max":
        return BasicRealizedProfitMax(cfg, initial_state=initial_state)
    elif cfg.type == "sharpe_ratio":
        return SharpeRatio(cfg, initial_state=initial_state)
    elif cfg.type == "sortino_ratio":
        return SortinoRatio(cfg, initial_state=initial_state)
    elif cfg.type == "calmar_ratio":
        return CalmarRatio(cfg, initial_state=initial_state)
    else:
        raise ValueError(f"Unknown reward function: {cfg.type}")
