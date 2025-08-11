import logging
from collections import defaultdict, deque
from enum import Enum

import numpy as np
import pandas as pd


class PositionType(str, Enum):
    LONG = "long"
    SHORT = "short"


class Position(np.ndarray):
    """
    Represents a position in the portfolio, backed by a numpy array.
    """

    COLS = [
        "symbol",
        "lot_size",
        "enter_price",
        "enter_date",
        "exit_date",
        "exit_price",
        "exit_size",
        "position_type",
    ]
    IDX_SYMBOL = 0
    IDX_LOT_SIZE = 1
    IDX_ENTER_DATE = 2
    IDX_ENTER_PRICE = 3
    IDX_EXIT_DATE = 4
    IDX_EXIT_PRICE = 5
    IDX_EXIT_SIZE = 6
    IDX_POSITION_TYPE = 7

    def __new__(
        cls, symbol: str, lot_size: float, enter_price: float, enter_date: pd.Timestamp
    ):
        data = [
            symbol,  # symbol
            lot_size,  # lot_size
            enter_date,  # enter_date
            enter_price,  # enter_price
            None,  # exit_date (default)
            None,  # exit_price (default)
            0.0,  # exit_size (default)
            1,  # position_type (default to LONG)
        ]
        obj = np.asarray(data, dtype=object).view(cls)
        return obj

    @property
    def symbol(self) -> str:
        return self[self.IDX_SYMBOL]

    @property
    def lot_size(self) -> float:
        return self[self.IDX_LOT_SIZE]

    @property
    def enter_price(self) -> float:
        return self[self.IDX_ENTER_PRICE]

    @property
    def exit_price(self) -> float | None:
        return self[self.IDX_EXIT_PRICE]

    @property
    def enter_date(self) -> pd.Timestamp:
        return self[self.IDX_ENTER_DATE]

    @property
    def exit_date(self) -> pd.Timestamp | None:
        return self[self.IDX_EXIT_DATE]

    @property
    def exit_size(self) -> float | None:
        return self[self.IDX_EXIT_SIZE]

    @property
    def position_type(self) -> PositionType:
        return (
            PositionType.LONG
            if self[self.IDX_POSITION_TYPE] > 0
            else PositionType.SHORT
        )

    def exit(
        self, exit_date: pd.Timestamp, price: float, exit_size: float | None = None
    ):
        self[self.IDX_EXIT_DATE] = exit_date
        self[self.IDX_EXIT_PRICE] = price
        if exit_size is not None:
            self[self.IDX_EXIT_SIZE] = exit_size
        return self.profit()

    def profit(self) -> float:
        if self.exit_price is None or self.enter_price is None:
            return 0.0
        return (
            (self.exit_price - self.enter_price) * self.exit_size
            if self.exit_size is not None
            else self.lot_size
        )

    def __repr__(self):
        return (
            "Position(symbol_idx=%s, size=%s, enter_price=%s, enter_date=%s, "
            "exit_date=%s, exit_price=%s, exit_size=%s, position_type=%s)"
            % (
                getattr(self, "symbol_idx", None),
                getattr(self, "size", None),
                self.enter_price,
                self.enter_date,
                self.exit_date,
                self.exit_price,
                self.exit_size,
                self.position_type,
            )
        )


class PositionManager:
    def __init__(self, symbols: list[str], max_lots: int = 1):
        self.symbols = symbols
        self.max_lots = max_lots
        self.df = pd.DataFrame(
            {
                "holdings": np.zeros(len(symbols), dtype=np.float32),
                "position_counts": np.zeros(len(symbols), dtype=np.int32),
                "rolling_profit": np.zeros(len(symbols), dtype=np.float32),
            },
            index=symbols,
        )
        self.positions: dict[str, deque[Position]] = {
            symbol: deque(maxlen=max_lots) for symbol in symbols
        }
        logging.info("Initializing PositionManager for symbols: %s", symbols)

    def __getitem__(self, key):
        """
        Allow access to the internal DataFrame using the [] accessor.
        """
        return self.df[key]

    def to_csv(self, path: str):
        """
        Save the positions to a CSV file.
        """
        data = []

        for symbol, positions in self.positions.items():
            for position in positions:
                data.append(
                    {
                        "symbol": symbol,
                        "size": position.size,
                        "enter_price": position.enter_price,
                        "enter_date": position.enter_date,
                        "exit_date": getattr(position, "exit_date", None),
                        "exit_price": getattr(position, "exit_price", None),
                    }
                )
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def reset(self):
        """
        Reset the position manager.
        """
        self.df["holdings"] = 0.0
        self.df["position_counts"] = 0
        self.df["rolling_profit"] = 0.0
        self.positions = {symbol: deque() for symbol in self.symbols}

    def append(self, df: pd.DataFrame):
        for sym, row in df.iterrows():
            self.positions[sym].append(
                Position(
                    symbol=str(sym),
                    lot_size=row["size"],
                    enter_price=row["price"],
                    enter_date=row["timestamp"],
                )
            )

    def exit_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exit positions based on the provided DataFrame.
        """
        step_profit = 0.0
        for sym, row in df.iterrows():
            remaining_lot = -row["size"]  # total to sell
            queue = self.positions[sym]
            len_positions = len(queue)
            profit = 0.0
            while remaining_lot > 0 and len(queue) > 0:
                max_to_take = min(queue[0][Position.IDX_EXIT_SIZE], remaining_lot)
                remaining_lot -= max_to_take
                queue[0][Position.IDX_EXIT_SIZE] -= max_to_take
                position = (
                    queue.popleft()
                    if queue[0][Position.IDX_EXIT_SIZE] == 0
                    else queue[0]
                )
                profit += position.exit(
                    exit_date=row["timestamp"],
                    price=row["price"],
                    exit_size=max_to_take,
                )
            step_profit += profit
            row.loc["profit"] = profit
            row.loc["size"] = row["size"] + remaining_lot
            row.loc["positions_counts"] = len(queue) - len_positions
            logging.debug("Exiting position for %s: profit=%s", sym, profit)
        return df

    def step(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """
        Step through the position manager.
        Return a set indicating if a position was exited and the profit from that position.
        """
        buy_mask = (df["size"] > 0) & (self.df["position_counts"] < self.max_lots)
        self.append(df[buy_mask])

        sell_mask = (df["size"] < 0) & (self.df["holdings"] > 0)
        exit_view = self.exit_positions(df[sell_mask])

        self.df.loc[buy_mask, "holdings"] += df.loc[buy_mask, "size"]
        self.df.loc[buy_mask, "position_counts"] += np.sign(df.loc[buy_mask, "size"])
        # df.loc[buy_mask, "size"] = df.loc[buy_mask, "size"]

        self.df.loc[sell_mask, "holdings"] += exit_view["size"]
        self.df.loc[sell_mask, "position_counts"] += np.sign(exit_view["size"])
        self.df.loc[sell_mask, "rolling_profit"] += exit_view["profit"]

        df.loc[sell_mask, "size"] = exit_view["size"]
        df.loc[~buy_mask & ~sell_mask, "size"] = 0.0

        return df, exit_view["profit"].sum()

    def nav(self, prices: pd.Series) -> float:
        """
        Calculate the net asset value (NAV) of the portfolio.
        """
        return (self.df["holdings"] * prices).sum()
