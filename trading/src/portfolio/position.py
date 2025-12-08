import logging
from collections import deque
from enum import Enum

import numpy as np
import pandas as pd

from trading.src.trade.trade_clients import TradingClient


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
            PositionType.LONG,  # position_type (default to LONG)
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
        return self[self.IDX_POSITION_TYPE]

    def exit(
        self, exit_date: pd.Timestamp, price: float, exit_size: float | None = None
    ):
        self[self.IDX_EXIT_DATE] = exit_date
        self[self.IDX_EXIT_PRICE] = price
        if exit_size is not None:
            self[self.IDX_EXIT_SIZE] = exit_size
        return self.profit()

    def profit(self) -> float:
        return (
            (self.exit_price if self.exit_price is not None else 0) - self.enter_price
        ) * (self.exit_size if self.exit_size is not None else 0)

    def __iter__(self):
        return iter(
            [
                self.symbol,
                self.lot_size,
                self.enter_price,
                self.enter_date,
                self.exit_date,
                self.exit_price,
                self.exit_size,
                self.position_type,
            ]
        )

    def __str__(self):
        return (
            "Position(symbol=%s, lot_size=%s, enter_price=%s, enter_date=%s, "
            "exit_date=%s, exit_price=%s, exit_size=%s, position_type=%s)"
            % (
                self.symbol,
                self.lot_size,
                self.enter_price,
                self.enter_date,
                self.exit_date,
                self.exit_price,
                self.exit_size,
                self.position_type,
            )
        )

    def __repr__(self):
        return (
            "Position(symbol=%s, size=%s, enter_price=%s, enter_date=%s, "
            "exit_date=%s, exit_price=%s, exit_size=%s, position_type=%s)"
            % (
                self.symbol,
                self.lot_size,
                self.enter_price,
                self.enter_date,
                self.exit_date,
                self.exit_price,
                self.exit_size,
                self.position_type,
            )
        )


class PositionManager:
    def __init__(
        self,
        symbols: list[str],
        max_lots: int | None = None,
        maintain_history: bool = True,
        initial_cash: float = 0.0,
    ):
        self.symbols = symbols
        self.max_lots: int | None = max_lots
        self.initial_cash = initial_cash
        self.total_value: float = initial_cash
        self.cash = initial_cash
        self.df = pd.DataFrame(
            {
                "holdings": np.zeros(len(symbols), dtype=np.float32),
                "position_counts": np.zeros(len(symbols), dtype=np.int32),
                "rolling_profit": np.zeros(len(symbols), dtype=np.float32),
            },
            index=symbols,
        )
        self.positions: dict[str, deque[Position]] = {
            symbol: deque(maxlen=self.max_lots) for symbol in symbols
        }
        self.history: list[Position] = []
        self.maintain_history = maintain_history
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
        for _, queue in self.positions.items():
            self.history.extend(queue)
        df = pd.DataFrame(self.history, columns=Position.COLS)
        df.to_csv(path, index=False)

    def reset(self):
        """
        Reset the position manager.
        """
        self.df["holdings"] = 0.0
        self.df["position_counts"] = 0
        self.df["rolling_profit"] = 0.0
        self.history = []
        self.positions = {symbol: deque() for symbol in self.symbols}

    def _append(self, df: pd.DataFrame):
        for sym, row in df.iterrows():
            self.positions[sym].append(
                Position(
                    symbol=str(sym),
                    lot_size=row["size"],
                    enter_price=row["price"],
                    enter_date=row["timestamp"],
                )
            )

    def _exit_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exit positions based on the provided DataFrame.
        """
        for row in df.itertuples(index=True):
            sym = row.Index
            remaining_lot = -row.size  # total to sell
            queue = self.positions[sym]
            profit = 0.0
            while remaining_lot > 0 and len(queue) > 0:
                max_to_take = min(queue[0][Position.IDX_LOT_SIZE], remaining_lot)
                remaining_lot -= max_to_take
                queue[0][Position.IDX_LOT_SIZE] -= max_to_take

                position = (
                    queue.popleft()
                    if queue[0][Position.IDX_LOT_SIZE] == 0
                    else queue[0]
                )

                position_profit = position.exit(
                    exit_date=row.timestamp,
                    price=row.price,
                    exit_size=max_to_take,
                )
                profit += position_profit
                if self.maintain_history:
                    self.history.append(position.copy())
                logging.debug("Exiting: %s profit=%s", position, position_profit)

            df.at[row.Index, "profit"] = profit
            df.at[row.Index, "size"] = row.size + remaining_lot

        return df

    def step(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """
        Step through the position manager.
        Return a set indicating if a position was exited and the profit from that position.
        """
        # Build a mask for buys that respects the max_lots limit and aligns indices to the incoming df
        if self.max_lots is None:
            lot_limit_mask = pd.Series(True, index=df.index)
        else:
            lot_limit_mask = (
                self.df["position_counts"].reindex(df.index).fillna(0) < self.max_lots
            )

        buy_mask = (df["size"] > 0) & lot_limit_mask
        self._append(df[buy_mask])

        # Build a mask for sells and align holdings to the incoming df's index
        sell_mask = (df["size"] < 0) & (
            self.df["holdings"].reindex(df.index).fillna(0) > 0
        )
        exit_view = self._exit_positions(df[sell_mask])

        # Use explicit symbol indexes for self.df updates to avoid misaligned boolean indexing
        buy_symbols = df.index[buy_mask]
        sell_symbols = df.index[sell_mask]

        self.df.loc[buy_symbols, "holdings"] += df.loc[buy_symbols, "size"]
        self.df.loc[buy_symbols, "position_counts"] += np.sign(
            df.loc[buy_symbols, "size"]
        )

        self.df.loc[sell_symbols, "holdings"] += exit_view.loc[sell_symbols, "size"]
        self.df.loc[sell_symbols, "position_counts"] += np.sign(
            exit_view.loc[sell_symbols, "size"]
        )
        self.df.loc[sell_symbols, "rolling_profit"] += exit_view.loc[
            sell_symbols, "profit"
        ]

        df.loc[sell_symbols, "size"] = exit_view.loc[sell_symbols, "size"]

        df.loc[~buy_mask & ~sell_mask, "size"] = 0.0

        return df, exit_view["profit"].sum()

    def nav(self, prices: pd.Series) -> float:
        """
        Calculate the net asset value (NAV) of the portfolio.
        """
        # Reindex the prices against the positions index to ensure alignment by values
        # (avoid joining on index names which can raise "cannot join with no overlapping index names").
        prices_aligned = prices.reindex(self.df.index).fillna(0.0)
        return (self.df["holdings"] * prices_aligned).sum()

    def net_value(self) -> float:
        """
        ! TODO implement this func I reckon
        Calculate the net value of the portfolio.
        """
        return 0.0


class LivePositionManager(PositionManager):
    """
    Position Manager derived class for implementing wrapper functionality to the underlying logic of the PositionManager.
    Additional logic to convert actual live portfolio positions into the PositionManager format, and thus perform that logic
    on the live data. Additionally Position Manager can be derived to perform other wrapper functionality on the underlying
    actions taken by the Position Manager logic.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        symbols: list[str],
        max_lots: int | None = None,
        maintain_history: bool = True,
        initial_cash: float = 0,
    ):
        super().__init__(symbols, max_lots, maintain_history, initial_cash)
        self.trading_client = trading_client

    def reset(self):
        raise NotImplementedError(
            "LivePositionManager reset not implemented yet. And it will not be!"
        )

    def step(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        df, profit = super().step(df)
        df, profit = self.trading_client.execute_trades(df)
        return df, profit
