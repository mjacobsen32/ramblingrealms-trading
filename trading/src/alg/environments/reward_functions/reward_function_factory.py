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
from trading.src.alg.environments.reward_functions.fast_profit_reward import (
    FastProfitReward,
    SimpleMomentumReward,
)


def reward_factory_method(
    cfg: RewardConfig, initial_state: np.ndarray
) -> RewardFunction:
    """
    Factory method to create a reward function instance based on the configuration.

    Recommended for training with FastTrainingEnv:
    - fast_profit_reward: Ultra-fast, minimal computation
    - simple_momentum_reward: Fast with percentage-based rewards

    Recommended for evaluation with StatefulTradingEnv:
    - basic_profit_max: Standard profit maximization
    - sharpe_ratio: Risk-adjusted returns
    - sortino_ratio: Downside risk focus
    - calmar_ratio: Max drawdown focus
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
    elif cfg.type == "fast_profit_reward":
        return FastProfitReward(cfg, initial_state=initial_state)
    elif cfg.type == "simple_momentum_reward":
        return SimpleMomentumReward(cfg, initial_state=initial_state)
    else:
        raise ValueError(f"Unknown reward function: {cfg.type}")
