import logging
from collections import defaultdict, deque
from enum import Enum

import numpy as np
import pandas as pd


class PositionType(str, Enum):
    LONG = "long"
    SHORT = "short"


class Positions:
    def __init__(self, symbols: list[str]):
        self.holdings = np.zeros(len(symbols), dtype=np.float32)
        self.position_counts = np.zeros(len(symbols), dtype=np.int32)
        self.rolling_profit = np.zeros(len(symbols), dtype=np.float32)
        self.symbols = symbols
        self.df = pd.DataFrame(
            index=pd.MultiIndex(
                levels=[[], symbols], codes=[[], []], names=["timestamp", "symbol"]
            ),
            columns=[
                "size",
                "enter_price",
                "enter_date",
                "exit_date",
                "exit_price",
                "position_type",
            ],
        )

    def reset(self):
        """
        Reset the position manager.
        """
        self.holdings.fill(0)
        self.position_counts.fill(0)
        self.rolling_profit.fill(0)
        self.df = pd.DataFrame(
            index=pd.MultiIndex(
                levels=[[], self.symbols], codes=[[], []], names=["timestamp", "symbol"]
            ),
            columns=[
                "size",
                "enter_price",
                "enter_date",
                "exit_date",
                "exit_price",
                "position_type",
            ],
        )

    def step(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.float64]:
        """
        Step through the position manager.
        Return a set indicating if a position was exited and the profit from that position.
        """

        buy_mask = df["size"] > 0
        buy_df = df[buy_mask][["size"]]

        buy_df["enter_price"] = df[buy_mask]["close"]
        buy_df["enter_date"] = df[buy_mask].index.get_level_values(0)
        buy_df["exit_date"] = None
        buy_df["exit_price"] = None
        buy_df["position_type"] = PositionType.LONG
        self.df = pd.concat([self.df, buy_df], axis=0)

        self.holdings += df["size"]
        self.position_counts += (df["size"] > 0).astype(int)
        # self._rolling_profit += df["size"] * (df["close"] - df["close"].shift(1)).fillna(0)

        profit = 0.0
        return buy_df, profit

    def nav(self, prices: pd.Series) -> float:
        """
        Calculate the net asset value (NAV) of the portfolio.
        """
        return (self.holdings * prices).sum()
