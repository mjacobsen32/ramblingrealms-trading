from trading.cli.alg.config import RewardConfig
from trading.src.alg.environments.reward_functions.basic_profit_max import (
    BasicProfitMax,
)


def factory_method(cfg: RewardConfig):
    """
    Factory method to create a reward function instance based on the configuration.
    """
    if cfg.type == "basic_profit_max":
        return BasicProfitMax(cfg)
    else:
        raise ValueError(f"Unknown reward function: {cfg.type}")
