import logging
import sys

from gymnasium.envs.registration import register

register(
    id="gymnasium_env/RR_TradingEnv-v0",
    entry_point="trading.src.alg.environments.trading_environment:TradingEnv",
)
