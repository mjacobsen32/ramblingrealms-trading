import pandas as pd
import vectorbt as vbt


class BackTesting:
    """
    Backtesting class for trading strategies using vectorbt.
    """

    def __init__(self, model, data, env):
        self.model = model
        self.data = data
        self.env = env
        self.records = []

    def run(self) -> vbt.Portfolio:
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

        self.pf = vbt.Portfolio.from_orders(
            close=close_df, size=size_df, init_cash=self.env.initial_cash
        )
        return self.pf

    def plot(self):
        """
        Plot the results of the backtest.
        """
        for c in self.data.tic.unique():
            self.pf[c].plot().show()

    def stats(self):
        """
        Return the statistics of the backtest.
        """
        return self.pf.stats()
