from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


class VectorizedPositionManager:
    """
    Manages positions in a vectorized manner using a pandas DataFrame as the underlying
    storage. The DataFrame uses a MultiIndex (timestamp, symbol) and tracks both open
    and closed positions.
    """

    COLUMNS = ["size", "enter_price", "enter_date", "exit_date", "exit_price"]

    def __init__(self, symbols: list[str] | None = None):
        self.positions = pd.DataFrame(columns=self.COLUMNS)
        self.positions["enter_date"] = pd.to_datetime(self.positions["enter_date"])  # type: ignore[index]
        self.positions["exit_date"] = pd.to_datetime(self.positions["exit_date"])  # type: ignore[index]
        self.positions.index = pd.MultiIndex.from_tuples(
            [], names=["timestamp", "symbol"]
        )
        self.total_profit: float = 0.0
        self.symbols: list[str] = list(symbols) if symbols is not None else []

    def set_symbols(self, symbols: list[str]) -> None:
        self.symbols = list(symbols)

    # --- Internal helpers -------------------------------------------------
    def _get_open_positions(self, symbol: str) -> pd.DataFrame:
        """Return open positions for a symbol ordered by enter_date (FIFO)."""
        if self.positions.empty:
            return self.positions.iloc[0:0]
        mask = self.positions.index.get_level_values("symbol") == symbol
        open_pos = self.positions.loc[mask]
        if open_pos.empty:
            return open_pos
        open_pos = open_pos[open_pos["exit_date"].isna()].copy()
        if open_pos.empty:
            return open_pos
        return open_pos.sort_values(by=["enter_date"])  # oldest first

    def _append_many(self, df: pd.DataFrame) -> None:
        if not df.empty:
            if "enter_date" in df:
                df.loc[:, "enter_date"] = pd.to_datetime(df["enter_date"])
            if "exit_date" in df:
                df.loc[:, "exit_date"] = pd.to_datetime(df["exit_date"])
            self.positions = pd.concat([self.positions, df[self.COLUMNS]])

    # --- Public API -------------------------------------------------------
    def step(self, orders: pd.DataFrame) -> Dict[str, Any]:
        """
        Vectorized processing of a batch of orders. The `orders` DataFrame must have a
        MultiIndex (timestamp, symbol) and columns: 'size' and 'close'.

        - Buys (size > 0) open new lots fully.
        - Sells (size < 0) close existing open lots in FIFO order, possibly partially.
        - The executed sizes are written back and returned.
        - Batch realized PnL is returned and accumulated into self.total_profit.
        """
        if orders.empty:
            return {"total_profit": 0.0, "executed_sizes": pd.Series(dtype=float)}

        if not isinstance(orders.index, pd.MultiIndex) or orders.index.nlevels != 2:
            raise ValueError("orders must have a MultiIndex (timestamp, symbol)")
        if not {"size", "close"}.issubset(orders.columns):
            raise ValueError("orders must contain 'size' and 'close' columns")

        executed_sizes = pd.Series(0.0, index=orders.index)
        batch_profit = 0.0

        # Buys in bulk
        buys = orders[orders["size"] > 0]
        if not buys.empty:
            buy_idx = buys.index
            buy_rows = pd.DataFrame(
                {
                    "size": buys["size"].astype(float).values,
                    "enter_price": buys["close"].astype(float).values,
                    "enter_date": buy_idx.get_level_values("timestamp").values,
                    "exit_date": pd.NaT,
                    "exit_price": np.nan,
                },
                index=pd.MultiIndex.from_arrays(
                    [buy_idx.get_level_values(0), buy_idx.get_level_values(1)],
                    names=["timestamp", "symbol"],
                ),
            )
            self._append_many(buy_rows)
            executed_sizes.loc[buys.index] = buys["size"].astype(float).values

        # Vectorized FIFO closes per symbol
        sells = orders[orders["size"] < 0]
        if not sells.empty:
            for _, sell_grp in sells.groupby(level="symbol", sort=False):
                sym_key = sell_grp.index.get_level_values("symbol").unique()[0]
                open_pos = self._get_open_positions(str(sym_key))
                if open_pos.empty:
                    executed_sizes.loc[sell_grp.index] = 0.0 - 0.0
                    continue

                lot_sizes = open_pos["size"].to_numpy(dtype=float)
                lot_enter_p = open_pos["enter_price"].to_numpy(dtype=float)
                lot_enter_d = pd.to_datetime(open_pos["enter_date"]).to_numpy()
                lot_index = open_pos.index

                sell_qty = -sell_grp["size"].to_numpy(dtype=float)
                sell_price = sell_grp["close"].to_numpy(dtype=float)
                sell_ts = pd.to_datetime(
                    sell_grp.index.get_level_values("timestamp")
                ).to_numpy()

                S = np.cumsum(lot_sizes)
                D = np.cumsum(sell_qty)
                S_prev = np.concatenate(([0.0], S[:-1]))
                D_prev = np.concatenate(([0.0], D[:-1]))

                min_SD = np.minimum(S[:, None], D[None, :])
                max_SpDp = np.maximum(S_prev[:, None], D_prev[None, :])
                F = (min_SD - max_SpDp).clip(min=0.0)

                if not np.any(F > 0):
                    executed_sizes.loc[sell_grp.index] = 0.0 - 0.0
                    continue

                exec_per_sell = F.sum(axis=0)
                closed_per_lot = F.sum(axis=1)

                profit_matrix = (sell_price[None, :] - lot_enter_p[:, None]) * F
                batch_profit += float(profit_matrix.sum())

                i_idx, j_idx = np.nonzero(F > 0)
                qty = F[i_idx, j_idx]
                closed_idx_ts = lot_index.get_level_values(0).to_numpy()[i_idx]
                closed_symbols = np.array([sym_key] * len(i_idx))
                closed_enter_price = lot_enter_p[i_idx]
                closed_enter_date = lot_enter_d[i_idx]
                closed_exit_date = sell_ts[j_idx]
                closed_exit_price = sell_price[j_idx]

                closed_rows = pd.DataFrame(
                    {
                        "size": qty,
                        "enter_price": closed_enter_price,
                        "enter_date": closed_enter_date,
                        "exit_date": closed_exit_date,
                        "exit_price": closed_exit_price,
                    },
                    index=pd.MultiIndex.from_arrays(
                        [closed_idx_ts, closed_symbols], names=["timestamp", "symbol"]
                    ),
                )
                self._append_many(closed_rows)

                new_sizes = lot_sizes - closed_per_lot
                remain_mask = new_sizes > 1e-12
                if remain_mask.any():
                    upd_index = lot_index[remain_mask]
                    upd_sizes = pd.Series(new_sizes[remain_mask], index=upd_index)
                    self.positions.loc[upd_index, "size"] = upd_sizes
                drop_index = lot_index[~remain_mask]
                if len(drop_index) > 0:
                    self.positions = self.positions.drop(index=drop_index)

                exec_series = pd.Series(-exec_per_sell, index=sell_grp.index)
                executed_sizes.loc[sell_grp.index] = exec_series.values

        self.total_profit += batch_profit
        orders.loc[:, "size"] = executed_sizes
        return {"total_profit": batch_profit, "executed_sizes": executed_sizes}

    def nav(self, prices: pd.Series) -> float:
        """
        Compute net asset value (NAV) of open positions using current prices.
        `prices` should be a pd.Series indexed by symbol with the latest price.

        NAV = sum_{symbols} (open_size(symbol) * price(symbol))
        """
        if self.positions.empty:
            return 0.0
        open_pos = self.positions[self.positions["exit_date"].isna()]
        if open_pos.empty:
            return 0.0
        held = open_pos.groupby(level="symbol")["size"].sum()
        aligned = held.reindex(prices.index).fillna(0.0)
        return float((aligned * prices.reindex(aligned.index)).fillna(0.0).sum())

    def positions_held(self) -> pd.Series:
        """Return total open shares held per symbol as a pd.Series indexed by symbol."""
        if self.positions.empty:
            return pd.Series(dtype=float)
        open_pos = self.positions[self.positions["exit_date"].isna()]
        if open_pos.empty:
            return pd.Series(dtype=float)
        return open_pos.groupby(level="symbol")["size"].sum().astype(float)

    def as_dataframe(self) -> pd.DataFrame:
        """Return the underlying positions DataFrame (copy)."""
        return self.positions.copy()

    # Compatibility helpers for Portfolio and tests
    def as_numpy(self, symbols: list[str] | None = None) -> np.ndarray:
        if symbols is None:
            symbols = self.symbols
        held = self.positions_held()
        return held.reindex(symbols).fillna(0.0).to_numpy(dtype=float)

    def positions_held_as_numpy(self, symbols: list[str] | None = None) -> np.ndarray:
        if symbols is None:
            symbols = self.symbols
        if self.positions.empty:
            return np.zeros(len(symbols), dtype=np.int32)
        open_pos = self.positions[self.positions["exit_date"].isna()]
        counts = open_pos.groupby(level="symbol").size()
        return counts.reindex(symbols).fillna(0).to_numpy(dtype=np.int32)

    def reset(self) -> None:
        """Reset all stored positions and PnL."""
        self.positions = pd.DataFrame(columns=self.COLUMNS)
        self.positions["enter_date"] = pd.to_datetime(self.positions["enter_date"])  # type: ignore[index]
        self.positions["exit_date"] = pd.to_datetime(self.positions["exit_date"])  # type: ignore[index]
        self.positions.index = pd.MultiIndex.from_tuples(
            [], names=["timestamp", "symbol"]
        )
        self.total_profit = 0.0

    def to_csv(self, path: str | Path) -> None:
        """Export all positions (open and closed) to CSV."""
        df = self.as_dataframe().reset_index()
        df.to_csv(path, index=False)


# Backward-compat alias for tests or external imports
VectorizedPosition = VectorizedPositionManager
