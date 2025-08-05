import logging

import numpy as np
import pandas as pd
import vectorbt as vbt
from rich import print as rprint

from trading.cli.alg.config import PortfolioConfig


class Portfolio:
    """
    Portfolio class for managing trading positions and cash flow.
    Very much Stateful
    """

    def __init__(self, cfg: PortfolioConfig, stock_dimension: int = 1):
        """
        Initialize the Portfolio
        """
        self.cfg = cfg
        self.initial_cash = cfg.initial_cash
        self.total_value = cfg.initial_cash
        self.cash = cfg.initial_cash
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
        price = self.df.pivot(columns="symbol", values="close")
        close = self.df.pivot(columns="symbol", values="close")
        size = self.df.pivot(columns="symbol", values="size")

        self.vbt_pf = vbt.Portfolio.from_orders(
            close=close,
            price=price,
            size=size,
            init_cash=100_000,
            log=True,
            cash_sharing=True,
        )
        return self.vbt_pf

    def state(self, timestamp: pd.Timestamp = None) -> np.ndarray:
        """
        Get the current state of the portfolio.
        Returns:
            pd.Series: [internal_cash, positions].
        """
        timestamp = timestamp or self.df.index[-1] if not self.df.empty else None
        positions = (
            self.df.loc[[timestamp]]["position"].values
            if not self.df.empty
            else np.zeros(self.stock_dimension)
        )
        return np.concatenate([[self.cash], positions])

    def net_value(self) -> float:
        """
        Calculate the net value of the portfolio.
        """
        return self.total_value

    def scale_actions(self, df: pd.DataFrame, prices: np.ndarray) -> np.ndarray:
        """
        Scale the actions to the portfolio size.
        Enfore environment constraints.
        """
        logging.debug(f"Raw Action: {df['size']}")

        """
            No short selling / no negative positions allowed (yet)
        """
        buy_mask = df["size"] > 0
        sell_mask = df["size"] < 0
        df["size"] = df.apply(
            lambda row: (
                max(
                    row["size"],
                    -(
                        self.df.groupby("symbol")["position"].last().get(row.name[1], 0)
                        if not self.df.empty
                        else 0
                    ),
                )
                if row["size"] < 0
                else row["size"]
            ),
            axis=1,
        )

        """
            Scale the actions to the portfolio size
        """

        buy_limit = self.cfg.trade_limit_percent * self.total_value

        # Compute per-trade cost
        buy_values = df["size"][buy_mask] * prices[buy_mask]

        # Cap each trade at buy_limit
        capped_values = np.minimum(buy_values, buy_limit)

        attempted_buy = capped_values.sum()

        if attempted_buy > self.cash:
            buy_limit = min(buy_limit, self.cash / buy_mask.sum())

        logging.debug(f"Buy Limit dollar amount: {buy_limit}")

        # Calculate the maximum shares allowed by hmax and buy_limit for each buy
        max_shares_hmax = self.cfg.hmax // prices[buy_mask]
        max_shares_buy_limit = buy_limit // prices[buy_mask]
        max_shares = np.minimum(max_shares_hmax, max_shares_buy_limit)

        # Clip the size to not exceed the minimum of both limits
        df.loc[buy_mask, "size"] = np.clip(df.loc[buy_mask, "size"], 0, max_shares)
        df["size"] = np.nan_to_num(df["size"])
        logging.debug(f"Scaled Actions: {df['size']}")

        return df["size"].values

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

    def step(self, prices, df) -> dict:
        """
        Take a step in the portfolio environment.
        """
        current_total_value = self.net_value()

        df["size"] = self.scale_actions(df, prices)
        self.update_position_batch(df=df)
        return {
            "scaled_actions": df["size"].values,
            "profit": self.net_value() - current_total_value,
        }

    @classmethod
    def load(cls, cfg: PortfolioConfig, file_path: str) -> "Portfolio":
        """
        Load backtesting results from a JSON file.
        """
        pf = vbt.Portfolio.load(file_path)
        logging.info(f"Loaded backtest results from {file_path}")
        ret = cls(cfg=cfg)
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
        logging.debug(f"Portfolio has been reset.\n{self}")

    def set_vbt(self, pf: vbt.Portfolio):
        """
        Set the portfolio object with backtest results.
        """
        self.vbt_pf = pf

    def plot(self):
        """
        Plot the results of the backtest.
        """
        self.as_vbt_pf().plot(
            title="Portfolio Backtest Results",
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
