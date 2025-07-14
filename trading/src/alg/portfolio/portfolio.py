import logging

import numpy as np
import pandas as pd
import vectorbt as vbt
from rich import print as rprint

#         self.sell_cost = (
#     [float(x) for x in cfg.sell_cost_pct]
#     if isinstance(cfg.sell_cost_pct, list)
#     else [float(cfg.sell_cost_pct)] * self.stock_dimension
# )
# self.buy_cost = (
#     [float(x) for x in cfg.buy_cost_pct]
#     if isinstance(cfg.buy_cost_pct, list)
#     else [float(cfg.buy_cost_pct)] * self.stock_dimension
# )


class Portfolio:
    """
    Portfolio class for managing trading positions and cash flow.
    Very much StateFULL
    """

    vbt_pf: vbt.Portfolio | None = None
    df = pd.DataFrame()

    def __init__(self, initial_cash: float = 100_000, stock_dimension: int = 1):
        """
        Initialize the Portfolio
        """
        self.initial_cash = initial_cash
        self.total_value = initial_cash
        self.cash = initial_cash
        self.nav = 0
        self.df = pd.DataFrame()
        self.vbt_pf: vbt.Portfolio | None = None
        self.stock_dimension = stock_dimension

    def as_vbt_pf(self) -> vbt.Portfolio:
        """
        Update the portfolio with new data.
        """
        if self.vbt_pf is not None:
            return self.vbt_pf

        self.df.reset_index(level="symbol", inplace=True)
        self.vbt_pf = vbt.Portfolio.from_orders(
            close=self.df["close"],
            size=self.df["size"],
            init_cash=100_000,
            log=True,
        )
        return self.vbt_pf

    def state(self, timestamp: pd.Timestamp) -> pd.Series:
        """
        Get the current state of the portfolio.
        Returns:
            pd.Series: [internal_cash, positions].
        """
        return pd.Series(
            [
                self.cash,
                (
                    self.df.loc[[timestamp]]["position"]
                    if not self.df.empty
                    else np.zeros(self.stock_dimension)
                ),
            ]
        )

    def net_value(self) -> float:
        """
        Calculate the net value of the portfolio.
        """
        return self.total_value

    def update_position_batch(self, df: pd.DataFrame):
        """
        Update positions for a batch of tickers at a given timestamp.
        """
        self.df = pd.concat([self.df, df.loc[:, ["close", "size"]]], axis=0)
        self.df["position"] = self.df.groupby("symbol")[
            "size"
        ].cumsum()  # shouldn't need to cumsum every time
        self.df["nav"] = self.df["position"] * self.df["close"]
        self.cash += -(df["size"] * df["close"]).sum()
        self.nav = self.df.groupby("symbol")["nav"].last().sum()
        self.total_value = self.cash + self.nav

    @classmethod
    def load(cls, file_path: str) -> "Portfolio":
        """
        Load backtesting results from a JSON file.
        """
        pf = vbt.Portfolio.load(file_path)
        logging.info(f"Loaded backtest results from {file_path}")
        ret = cls()
        ret.vbt_pf = pf
        return ret

    def save(self, file_path: str):
        logging.info(f"Saving backtest results to {file_path}")
        self.as_vbt_pf().save(file_path)

    def reset(self):
        """
        Reset the portfolio to an empty state.
        """
        self.initial_cash = self.initial_cash
        self.total_value = self.initial_cash
        self.cash = self.initial_cash
        self.nav = 0
        self.df = pd.DataFrame()
        self.vbt_pf: vbt.Portfolio | None = None
        logging.info(f"Portfolio has been reset.\n{self}")

    def set_vbt(self, pf: vbt.Portfolio):
        """
        Set the portfolio object with backtest results.
        """
        self.vbt_pf = pf

    def plot(self):
        """
        Plot the results of the backtest.
        """
        # for asset in self.as_vbt_pf().assets().columns.tolist():
        #     print(self.as_vbt_pf().subplots) available subplots
        self.as_vbt_pf().plot(
            title="Portfolio",
            subplots=[
                "cash",
                "asset_flow",
                "trades",
                "trade_pnl",
                "cum_returns",
                "orders",
            ],
        ).show()

    def stats(self):
        """
        Return the statistics of the backtest.
        """
        return self.as_vbt_pf().stats()

    def orders(self):
        """
        Return the orders of the backtest.
        """
        return self.as_vbt_pf().orders

    def __repr__(self) -> str:
        """
        String representation of the Portfolio.
        """
        return f"Portfolio(initial_cash={self.initial_cash}, total_value={self.total_value}, cash={self.cash}, nav={self.nav}, df_shape={self.df.shape})"
