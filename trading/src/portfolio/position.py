import datetime
import json
import logging
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
from alpaca.broker.requests import MarketOrderRequest
from alpaca.trading.models import Position as AlpacaPosition
from fastparquet.converted_types import nullable
from pydantic import BaseModel

if TYPE_CHECKING:
    from trading.src.trade.trade_clients import TradingClient

portfolio_schema = pa.schema(
    [
        pa.field("net_value", pa.float64(), nullable=False),
        pa.field("cash", pa.float64(), nullable=False),
        pa.field("pnl", pa.float64(), nullable=False),
        pa.field("pnl_pct", pa.float64(), nullable=False),
        pa.field("date", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("rolling_pnl", pa.float64(), nullable=False),
        pa.field("rolling_pnl_pct", pa.float64(), nullable=False),
    ]
)


class PortfolioStats(BaseModel):
    COLS: ClassVar[list[str]] = [
        "net_value",
        "cash",
        "pnl",
        "pnl_pct",
        "date",
        "rolling_pnl",
        "rolling_pnl_pct",
    ]
    net_value: float = 0.0
    cash: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    rolling_pnl: float = 0.0
    rolling_pnl_pct: float = 0.0
    date: datetime.datetime = datetime.datetime.utcnow()

    def __iter__(self):
        return iter(
            [
                self.net_value,
                self.cash,
                self.pnl,
                self.pnl_pct,
                self.date,
                self.rolling_pnl,
                self.rolling_pnl_pct,
            ]
        )

    def to_row(self) -> list:
        return [
            self.net_value,
            self.cash,
            self.pnl,
            self.pnl_pct,
            self.date,
            self.rolling_pnl,
            self.rolling_pnl_pct,
        ]


class PositionType(str, Enum):
    LONG = "long"
    SHORT = "short"


class PositionEncoder(json.JSONEncoder):
    def default(self, obj):
        return {
            "symbol": obj.symbol,
            "lot_size": obj.lot_size,
            "enter_price": obj.enter_price,
            "enter_date": obj.enter_date.isoformat(),
            "exit_date": (
                obj.exit_date.isoformat() if obj.exit_date is not None else None
            ),
            "exit_price": obj.exit_price,
            "exit_size": obj.exit_size,
            "position_type": obj.position_type,
        }


def PositionDecoder(dct):
    return Position(
        symbol=dct["symbol"],
        lot_size=dct["lot_size"],
        enter_price=dct["enter_price"],
        enter_date=pd.Timestamp(dct["enter_date"]),
        exit_date=(
            pd.Timestamp(dct["exit_date"]) if dct.get("exit_date") is not None else None
        ),
        exit_price=dct.get("exit_price"),
        exit_size=dct.get("exit_size"),
        position_type=PositionType(dct["position_type"]),
    )


positions_schema = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("lot_size", pa.float64(), nullable=False),
        pa.field("enter_date", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("enter_price", pa.float64(), nullable=False),
        pa.field("exit_date", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("exit_price", pa.float64(), nullable=False),
        pa.field("exit_size", pa.float64(), nullable=False),
        pa.field("position_type", pa.string(), nullable=False),
    ]
)


class Position(np.ndarray):
    """
    Represents a position in the portfolio, backed by a numpy array.
    TODO swap out in favor of alpacas Position type, if possible, however speed of np.ndarray is preferred.
    """

    COLS = [
        "symbol",
        "lot_size",
        "enter_date",
        "enter_price",
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

    def to_row(self) -> list:
        """
        This method is for completeness and consistency, as it derives from np.ndarray, it is alread a list-like object.

        :param self: Description
        :return: Description
        :rtype: list[Any]
        """
        return [
            self.symbol,
            self.lot_size,
            self.enter_date,
            self.enter_price,
            self.exit_date,
            self.exit_price,
            self.exit_size,
            self.position_type.value,
        ]

    @classmethod
    def from_alpaca_position(cls, alpaca_position: AlpacaPosition) -> "Position":
        return cls(
            symbol=alpaca_position.symbol,
            lot_size=float(
                alpaca_position.qty_available
                if alpaca_position.qty_available
                else alpaca_position.qty
            ),
            enter_price=float(alpaca_position.avg_entry_price),
            enter_date=None,
            exit_date=None,
            exit_price=None,
            exit_size=None,
            position_type=(
                PositionType.LONG
                if alpaca_position.side == "long"
                else PositionType.SHORT
            ),
        )

    def __new__(
        cls,
        symbol: str,
        lot_size: float,
        enter_price: float,
        enter_date: pd.Timestamp | None = None,
        exit_date: pd.Timestamp | None = None,
        exit_price: float | None = None,
        exit_size: float | None = None,
        position_type: PositionType = PositionType.LONG,
    ):
        data = [
            symbol,  # symbol
            lot_size,  # lot_size
            enter_date,  # enter_date
            enter_price,  # enter_price
            exit_date,  # exit_date
            exit_price,  # exit_price
            exit_size,  # exit_size
            position_type,  # position_type
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
    def enter_date(self) -> pd.Timestamp | None:
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
                self.enter_date,
                self.enter_price,
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
        df: pd.DataFrame | None = None,
        positions: dict[str, deque[Position]] | None = None,
        cash: float | None = None,
    ):
        self.symbols = symbols
        self.max_lots: int | None = max_lots
        self._initial_cash = float(initial_cash)
        self._cash = float(initial_cash if cash is None else cash)
        if df is None:
            self.df = pd.DataFrame(
                {
                    "holdings": np.zeros(len(symbols), dtype=np.float32),
                    "position_counts": np.zeros(len(symbols), dtype=np.int32),
                    "rolling_profit": np.zeros(len(symbols), dtype=np.float32),
                    "price": np.zeros(len(symbols), dtype=np.float32),
                },
                index=symbols,
            )
        else:
            self.df = df
        if positions is not None:
            self.open_positions = positions
        else:
            self.open_positions = {
                symbol: deque(maxlen=self.max_lots) for symbol in symbols
            }
        self.closed_positions: list[Position] = []
        self.maintain_history = maintain_history
        logging.info("Initializing PositionManager with symbols: %s", symbols)

    def __getitem__(self, key):
        """
        Allow access to the internal DataFrame using the [] accessor.
        """
        return self.df[key]

    def to_csv(self, path: str):
        """
        Save the positions to a CSV file.
        """
        for _, queue in self.open_positions.items():
            self.closed_positions.extend(queue)
        df = pd.DataFrame(self.closed_positions, columns=Position.COLS)
        df.to_csv(path, index=False)

    def reset(self):
        """
        Reset the position manager.
        """
        self.df["holdings"] = 0.0
        self.df["position_counts"] = 0
        self.df["rolling_profit"] = 0.0
        self.closed_positions = []
        self.open_positions = {symbol: deque() for symbol in self.symbols}

    @classmethod
    def from_client(
        cls,
        trading_client: "TradingClient",
        symbols: list[str],
        max_lots: int | None = None,
        maintain_history: bool = True,
        initial_cash: float = 0.0,
        initial_prices: np.ndarray | None = None,
    ) -> "PositionManager":
        logging.info("Loading positions from trading client.")

        manager = object.__new__(cls)
        positions = trading_client.positions
        cash = trading_client.account.cash
        df = pd.DataFrame(
            {
                "holdings": np.zeros(len(symbols), dtype=np.float32),
                "position_counts": np.zeros(len(symbols), dtype=np.int32),
                "rolling_profit": np.zeros(len(symbols), dtype=np.float32),
                "price": (
                    initial_prices
                    if initial_prices is not None
                    else np.zeros(len(symbols), dtype=np.float32)
                ),
            },
            index=symbols,
        )
        # Initialize core attributes first so methods that rely on them can run
        PositionManager.__init__(
            self=manager,
            symbols=symbols,
            max_lots=max_lots,
            maintain_history=maintain_history,
            initial_cash=initial_cash,
            df=df,
            positions=None,  # create fresh deques and then populate them with to_df
            cash=cash,
        )
        # Populate the instance's data structures using client positions
        manager.to_df(positions, df)
        logging.info("PositionManager state: %s", manager.df)
        return manager

    def to_df(
        self, client_positions: dict[str, deque["Position"]], df: pd.DataFrame
    ) -> None:
        """Populate this PositionManager from a mapping of positions (e.g. from a TradingClient)."""
        for sym, pos_list in client_positions.items():
            if sym not in self.open_positions:
                logging.warning(
                    "Symbol %s not in PositionManager symbols; skipping", sym
                )
                continue
            total = 0.0
            for pos in pos_list:
                self.open_positions[sym].append(pos)
                total += float(pos.lot_size)
            df.loc[sym, "holdings"] = total
            df.loc[sym, "position_counts"] = len(pos_list)

    def _append(self, df: pd.DataFrame):
        for sym, row in df.iterrows():
            self.open_positions[sym].append(
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
        ! Removing the whole lots thing would reduce complexity and speed. AKA single "position" per symbol with size attribute.
        """
        for row in df.itertuples(index=True):
            sym = row.Index
            remaining_lot = -row.size  # total to sell
            queue = self.open_positions[sym]
            profit = 0.0
            while remaining_lot > 0 and len(queue) > 0:
                max_to_take = min(queue[0][Position.IDX_LOT_SIZE], remaining_lot)
                remaining_lot -= max_to_take
                queue[0][Position.IDX_LOT_SIZE] -= max_to_take

                if queue[0][Position.IDX_LOT_SIZE] == 0:
                    position = queue.popleft()
                    self.df.at[sym, "position_counts"] -= 1
                else:
                    position = queue[0]

                position_profit = position.exit(
                    exit_date=row.timestamp,
                    price=row.price,
                    exit_size=max_to_take,
                )
                profit += position_profit
                if self.maintain_history:
                    self.closed_positions.append(position.copy())
                logging.debug("Exiting: %s profit=%s", position, position_profit)

            df.at[row.Index, "profit"] = profit
            df.at[row.Index, "size"] = row.size + remaining_lot

        return df

    def step(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, float, list[MarketOrderRequest]]:
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

        # Use explicit symbol indexes for self.df updates to avoid misaligned boolean indexing
        buy_symbols = df.index[buy_mask]
        sell_symbols = df.index[sell_mask]

        df.loc[sell_mask] = self._exit_positions(df[sell_mask])

        df.loc[~buy_mask & ~sell_mask, "size"] = 0.0

        self._cash -= (df["size"] * df["price"]).sum()
        self.df["holdings"] += df["size"]
        self.df.loc[buy_symbols, "position_counts"] += np.sign(
            df.loc[buy_symbols, "size"]
        )

        self.df.loc[sell_symbols, "rolling_profit"] += df.loc[sell_symbols, "profit"]

        self.df["price"] = df["price"]

        return df, df.loc[sell_symbols, "profit"].sum(), []

    def available_cash(self) -> float:
        """
        Return the available cash of the portfolio.
        """
        return self._cash

    def initial_cash(self) -> float:
        """
        Return the initial cash of the portfolio.
        """
        return self._initial_cash

    def nav(self, prices: pd.Series | None = None) -> float:
        """
        Calculate the net asset value (NAV) of the portfolio.
        """
        if prices is not None:
            self.df["price"] = prices
        return (self.df["holdings"] * self.df["price"]).sum()

    def net_value(self, prices: pd.Series | None = None) -> float:
        return self.available_cash() + self.nav(prices)


class LivePositionManager(PositionManager):
    """
    Position Manager derived class for implementing wrapper functionality to the underlying logic of the PositionManager.
    Additional logic to convert actual live portfolio positions into the PositionManager format, and thus perform that logic
    on the live data. Additionally Position Manager can be derived to perform other wrapper functionality on the underlying
    actions taken by the Position Manager logic.
    """

    def __init__(
        self,
        trading_client: "TradingClient",
        symbols: list[str],
        max_lots: int | None = None,
        maintain_history: bool = True,
        initial_cash: float = 0,
        initial_prices: np.ndarray | None = None,
    ):
        # Create a fully initialized PositionManager and copy its state onto this instance
        manager = PositionManager.from_client(
            trading_client=trading_client,
            symbols=symbols,
            max_lots=max_lots,
            maintain_history=maintain_history,
            initial_cash=initial_cash,
            initial_prices=initial_prices,
        )
        # copy manager attributes (including _cash) into this LivePositionManager instance
        self.__dict__.update(manager.__dict__)
        self.pf_history: list[PortfolioStats] = []
        self.trading_client = trading_client

    def __del__(self):
        self.trading_client.close(
            closed_positions=self.closed_positions,
            open_positions=self.open_positions,
            pf_history=self.pf_history,
            cash=self._cash,
        )

    def reset(self):
        pass  # Override to do nothing; live positions are managed externally

    def step(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, float, list[MarketOrderRequest]]:
        df, profit, _ = super().step(df)
        # I think the pnl_pct should be from the previous net_worth
        stats = PortfolioStats(
            net_value=self.net_value(),
            cash=self.available_cash(),
            pnl=df["profit"].sum(),
            date=df["timestamp"].max(),
            pnl_pct=(
                df["profit"].sum() / self.net_value() if self.net_value() != 0 else 0
            ),
            rolling_pnl=self.net_value() - self._initial_cash,
            rolling_pnl_pct=(
                self.net_value() / self._initial_cash - 1
                if self._initial_cash != 0
                else 0
            ),
        )
        self.pf_history.append(stats)
        df, profit, orders = self.trading_client.execute_trades(actions=df)
        return df, profit, orders
