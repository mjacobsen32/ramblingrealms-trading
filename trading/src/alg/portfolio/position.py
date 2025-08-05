from collections import defaultdict, deque
from enum import Enum

import numpy as np
import pandas as pd


class PositionType(str, Enum):
    LONG = "long"
    SHORT = "short"


class Position:
    """
    Represents a position in the portfolio.
    """

    def __init__(
        self, symbol: str, size: float, price: float, enter_date: pd.Timestamp
    ):
        self.symbol = symbol
        self.position_type = PositionType.LONG if size > 0 else PositionType.SHORT
        self.size = size
        self.enter_price = price
        self.enter_date = enter_date
        self.exit_date = None

    def exit(
        self, exit_date: pd.Timestamp, price: float, partial_exit: float | None = None
    ):
        """
        Mark the position as exited.
        """
        self.exit_date = exit_date
        self.exit_price = price
        self.partial_exit = partial_exit
        return self.profit()

    def profit(self) -> float:
        """
        Calculate the profit from the position.
        """
        if self.exit_date is None:
            return 0.0
        return (self.exit_price - self.enter_price) * (
            self.size if self.partial_exit is None else self.partial_exit
        )

    def __repr__(self):
        return f"Position(symbol={self.symbol}, size={self.size}, enter_price={self.enter_price}, enter_date={self.enter_date}, exit_date={self.exit_date})"


class PositionView:
    """
    A view of positions for a specific symbol.
    """

    def __init__(self):
        self.size = 0.0
        self.rolling_return = 0.0

    def __iadd__(self, size: float):
        self.size += size
        return self

    def __isub__(self, size: float):
        self.size -= size
        return self

    def __repr__(self):
        return f"PositionView(size={self.size}, rolling_return={self.rolling_return})"


class PositionManager(defaultdict):
    def __init__(self, symbols: list[str], maintain_history: bool = True):
        super().__init__(deque)
        self.position_view = defaultdict(PositionView)
        self.symbols = symbols
        self.history: dict[str, deque[Position]] | None

        if maintain_history:
            self.history = {symbol: deque() for symbol in symbols}
        else:
            self.history = None
        for symbol in symbols:
            self[symbol] = deque()
            self.position_view[symbol] = PositionView()

    def reset(self):
        """
        Reset the position manager.
        """
        self.clear()
        self.position_view.clear()
        if self.history is not None:
            self.history.clear()

    def as_numpy(self) -> np.ndarray:
        return np.array(
            [self.position_view[symbol].size for symbol in self.symbols],
            dtype=np.float32,
        )

    def __repr__(self):
        return f"PositionManager(positions={dict(self)}, history={self.history is not None})"

    def __getitem__(self, item):
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __len__(self):
        return sum(len(positions) for positions in self.values())

    def step(
        self, symbol: str, date: pd.Timestamp, size: float, price: float
    ) -> tuple[bool, float]:
        """
        Step through the position manager.
        Return a set indicating if a position was exited and the profit from that position.
        """
        profit = 0.0
        self.position_view[symbol].size += size
        if symbol not in self:
            self[symbol] = deque()
        if size > 0.0:
            # BUY Push new position
            self[symbol].append(Position(symbol, size, price, date))
        elif size < 0.0:
            # SELL Close as many positions as needed
            remaining_shares = -size
            while remaining_shares > 0 and len(self[symbol]) > 0:
                if remaining_shares >= self[symbol][0].size:
                    # Close the entire position
                    position = self[symbol].popleft()
                    remaining_shares -= position.size
                    profit += position.exit(date, price)
                    if self.history:
                        self.history[symbol].append(position)
                else:
                    # Close a partial position
                    position = self[symbol][0]
                    position.size -= remaining_shares
                    profit += position.exit(date, price, partial_exit=remaining_shares)
                    remaining_shares = 0.0
                    if self.history:
                        self.history[symbol].append(position)
            self.position_view[symbol].rolling_return += profit
            return True, profit
        return False, profit

    def append(self, symbol: str, position: Position):
        """
        Append a new position to the manager.
        """
        if symbol not in self:
            self[symbol] = deque()
        self[symbol].append(position)

    def popleft(self, symbol: str) -> Position | None:
        """
        Pop the oldest position from the manager.
        """
        if symbol in self and self[symbol]:
            return self[symbol].popleft()
        return None

    def nav(self, prices: pd.Series) -> float:
        """
        Calculate the net asset value (NAV) of the portfolio.
        """
        return (self.as_numpy() * prices).sum()
