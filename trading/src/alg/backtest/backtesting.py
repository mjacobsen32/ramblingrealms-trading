import pickle

import pandas as pd
import vectorbt as vbt
from rich import print as rprint

from trading.cli.alg.config import BackTestConfig
from trading.src.alg.portfolio.portfolio import Portfolio


class BackTesting:
    """
    Backtesting class for trading strategies using vectorbt.
    """

    def __init__(self, model, data, env, backtest_config: BackTestConfig):
        self.model = model
        self.data = data
        self.env = env
        self.backtest_config = backtest_config

    def run(self) -> Portfolio:
        """
        Run the backtest using the provided model and environment.
        """
        obs, _ = self.env.reset()
        terminated, truncated = False, False

        while not terminated and not truncated:
            action, _states = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)

        if self.backtest_config.save_results:
            self.env.pf.save(str(self.backtest_config.results_path))

        return self.env.pf
