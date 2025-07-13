import logging
import sys

from gymnasium.envs.registration import register
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


debug_file_handler = logging.FileHandler("./logs/rr_trading.log")
debug_file_handler.setLevel(logging.DEBUG)

info_stdout_handler = logging.StreamHandler(sys.stdout)
info_stdout_handler.setLevel(logging.INFO)

register(
    id="gymnasium_env/RR_TradingEnv-v0",
    entry_point="trading.src.alg.environments.trading_environment:TradingEnv",
)
