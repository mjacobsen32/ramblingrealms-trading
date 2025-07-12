import pickle

import pandas as pd
import vectorbt as vbt
from rich import print as rprint

from trading.cli.alg.config import BackTestConfig


class Portfolio:
    pf: vbt.Portfolio

    @classmethod
    def load(cls, file_path: str) -> "Portfolio":
        """
        Load backtesting results from a JSON file.
        """
        pf = vbt.Portfolio.load(file_path)
        rprint(f"[blue]Loaded backtest results from {file_path}[/blue]")
        ret = cls()
        ret.pf = pf
        return ret

    def save(self, file_path: str):
        rprint(f"[blue]Saving backtest results to {file_path}[/blue]")
        self.pf.save(file_path)

    def set_pf(self, pf: vbt.Portfolio):
        """
        Set the portfolio object with backtest results.
        """
        self.pf = pf

    def plot(self):
        """
        Plot the results of the backtest.
        """
        for asset in self.pf.assets().columns.tolist():
            self.pf[asset].plot(title=asset).show()

    def stats(self):
        """
        Return the statistics of the backtest.
        """
        return self.pf.stats()


class BackTesting:
    """
    Backtesting class for trading strategies using vectorbt.
    """

    def __init__(self, model, data, env, backtest_config: BackTestConfig):
        self.model = model
        self.data = data
        self.env = env
        self.backtest_config = backtest_config
        self.pf = Portfolio()

    def get_portfolio(self) -> Portfolio:
        """
        Get the portfolio object containing backtest results.
        """
        return self.pf

    def run(self) -> Portfolio:
        """
        Run the backtest using the provided model and environment.
        """
        obs, _ = self.env.reset()
        terminated, truncated = False, False

        while not terminated and not truncated:
            action, _states = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)

        close_df = self.env.data.pivot(index="timestamp", columns="tic", values="close")
        size_df = self.env.data.pivot(index="timestamp", columns="tic", values="size")

        self.pf.set_pf(
            pf=vbt.Portfolio.from_orders(
                close=close_df, size=size_df, init_cash=self.env.initial_cash
            )
        )

        if self.backtest_config.save_results:
            rprint(
                f"[blue]Saving backtest results to {self.backtest_config.results_path.as_path()}[/blue]"
            )
            self.pf.save(str(self.backtest_config.results_path))

        return self.pf
