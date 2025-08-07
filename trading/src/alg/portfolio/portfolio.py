import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from plotly import io as pio
from rich import print as rprint
from vectorbt import _typing as tp

from trading.cli.alg.config import PortfolioConfig, ProjectPath, SellMode, TradeMode
from trading.src.alg.portfolio.position import Position, PositionManager


class Portfolio:
    """
    Portfolio class for managing trading positions and cash flow.
    Very much Stateful
    todo: move action to "action" and "size" is the scaled_action
    """

    def __init__(self, cfg: PortfolioConfig, symbols: list[str]):
        """
        Initialize the Portfolio
        """
        self.cfg = cfg
        self.initial_cash = cfg.initial_cash
        self.total_value: float = cfg.initial_cash
        self.cash = cfg.initial_cash
        self.nav: float = 0.0
        self.df = pd.DataFrame()
        self.vbt_pf: vbt.Portfolio | None = None
        self.symbols = symbols
        self._positions = PositionManager(symbols=symbols, maintain_history=True)

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

    def state(self) -> np.ndarray:
        """
        Get the current state of the portfolio.
        Returns:
            pd.Series: [internal_cash, positions].
        """
        return np.concatenate([[self.cash], self._positions.as_numpy()])

    def net_value(self) -> float:
        """
        Calculate the net value of the portfolio.
        """
        return self.total_value

    def enforce_trade_rules(
        self,
        df: pd.DataFrame,
        prices: np.ndarray,
        sell_mode: SellMode = SellMode.CONTINUOUS,
    ) -> np.ndarray:
        """
        Enforce trade rules on the actions.
        From trade sizes to actual trade sizes based on configuration and current portfolio state.
        """
        logging.debug(f"Raw Action: {df['size']}")

        max_positions_mask = (
            self._positions.positions_held_as_numpy() < self.cfg.max_positions
            if self.cfg.max_positions is not None
            else True
        )

        buy_mask = max_positions_mask & (df["size"] > 0)
        sell_mask = df["size"] < 0
        """
            No short selling / no negative positions allowed (yet)
        """

        positions = self._positions.as_numpy()
        if sell_mode == SellMode.CONTINUOUS:
            # Sell proportionally based on signal strength
            df.loc[sell_mask, "size"] = np.clip(
                df.loc[sell_mask, "size"], -positions[sell_mask], 0
            )
        elif sell_mode == SellMode.DISCRETE:
            # Sell entire position if sell action is triggered
            df.loc[sell_mask, "size"] = -positions[sell_mask]

        """
            Scale the actions to the portfolio value
        """

        buy_limit = self.cfg.trade_limit_percent * self.total_value

        # Compute per-trade cost
        buy_values = df.loc[buy_mask, "size"] * prices[buy_mask]

        # Cap each trade at buy_limit
        capped_values = np.minimum(buy_values, buy_limit)

        attempted_buy = capped_values.sum()

        if attempted_buy > self.cash:
            buy_limit = min(buy_limit, self.cash // len(buy_mask))

        logging.debug(f"Buy Limit dollar amount: {buy_limit}")

        # Calculate the maximum shares allowed by hmax and buy_limit for each buy
        max_shares_hmax = self.cfg.hmax // prices[buy_mask]
        max_shares_buy_limit = buy_limit // prices[buy_mask]
        max_shares = np.minimum(max_shares_hmax, max_shares_buy_limit)

        # Clip the size to not exceed the minimum of both limits
        df.loc[buy_mask, "size"] = np.clip(df.loc[buy_mask, "size"], 0, max_shares)
        df.loc[~buy_mask & ~sell_mask, "size"] = 0.0
        df["size"] = np.nan_to_num(df["size"])
        logging.debug(f"Scaled Actions: {df['size']}")

        return df["size"].values

    def scale_actions(
        self,
        df: pd.DataFrame,
        prices: np.ndarray,
        trade_mode: TradeMode = TradeMode.CONTINUOUS,
    ) -> np.ndarray:
        """
        Scale the actions to the maximum position size.
        From signal strength to a desired trade size
        """
        # Find actions above threshold
        above_thresh = df["action"].abs() > self.cfg.action_threshold

        # Calculate max shares allowed by hmax and trade_limit
        max_trade_value = min(
            self.cfg.hmax, self.cfg.trade_limit_percent * self.net_value()
        )
        max_shares = max_trade_value // prices

        if trade_mode == TradeMode.DISCRETE:
            df.loc[above_thresh, "size"] = df.loc[above_thresh, "action"].astype(
                np.int64
            ) * max_shares[above_thresh].astype(np.int64)
        elif trade_mode == TradeMode.CONTINUOUS:
            df.loc[above_thresh, "size"] = np.round(
                df.loc[above_thresh, "action"].astype(np.int64)
                * max_shares[above_thresh].astype(np.int64)
            )

        df.loc[~above_thresh, "size"] = 0.0

        return df["size"].values

    def update_position_batch(self, df: pd.DataFrame) -> float:
        """
        Update positions for a batch of tickers at a given timestamp.
        """
        step_profit = 0.0
        trade_mask = df["size"] != 0

        for multi_index, row in df[trade_mask].iterrows():
            _, actual_size, profit = self._positions.step(
                multi_index[1], multi_index[0], row["size"], row["close"]
            )
            step_profit += profit
            df.loc[multi_index, "size"] = actual_size

        self.df = pd.concat([self.df, df.loc[:, ["close", "size"]]], axis=0)
        self.cash = self.cash + -(df["size"] * df["close"]).sum()
        self.nav = self._positions.nav(df["close"])
        self.total_value = self.cash + self.nav

        return step_profit

    def step(
        self, prices: np.ndarray, df: pd.DataFrame, normalized_actions: bool = False
    ) -> dict:
        """
        Scale the actions and take a step in the portfolio environment.
        1. scale actions to trade size
        2. enforce trade limits
        3. update positions
        """
        if normalized_actions:
            df["size"] = self.scale_actions(df, prices, self.cfg.trade_mode)

        df["size"] = self.enforce_trade_rules(df, prices, self.cfg.sell_mode)
        step_profit = self.update_position_batch(df=df)
        return {
            "scaled_actions": df["size"].values,
            "profit": step_profit,
        }

    @classmethod
    def load(cls, cfg: PortfolioConfig, file_path: Path) -> "Portfolio":
        """
        Load backtesting results from a JSON file.
        """
        pf = vbt.Portfolio.load(str(file_path))
        logging.info(f"Loaded backtest results from {file_path}")
        ret = cls(cfg=cfg, symbols=[])
        ret.vbt_pf = pf
        return ret

    def save(self, file_path: str):
        logging.info(f"Saving backtest results to {file_path}")
        self.as_vbt_pf().save(file_path)

    def save_plots(self, backtest_dir: Path):
        plots = []
        paths = []
        for p in self._get_plots():
            title = p.layout.title.text.replace(" ", "_").lower()
            path = backtest_dir / "plots" / f"{title}.svg"
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            p.update_layout(width=1200, height=2400)
            plots.append(p)
            paths.append(path)
        pio.write_images(plots, paths, format="svg", scale=2, width=1200, height=2400)

    def reset(self):
        """
        Reset the portfolio to an empty state.
        """
        self.initial_cash = self.initial_cash
        self.total_value = self.initial_cash
        self.cash = self.initial_cash
        self.nav = 0.0
        self.df = pd.DataFrame()
        self.vbt_pf = None
        self._positions.reset()
        logging.debug(f"Portfolio has been reset.\n{self}")

    def set_vbt(self, pf: vbt.Portfolio):
        """
        Set the portfolio object with backtest results.
        """
        self.vbt_pf = pf

    def _get_plots(self) -> list[tp.BaseFigure]:
        """
        Get the plots for the portfolio.
        """
        vbt_pf = self.as_vbt_pf()
        plots: list[tp.BaseFigure] = []
        pio.renderers.default = "browser"
        logging.info(
            f"Generating portfolio plots...\nRendering with: {pio.renderers.default}"
        )
        p = vbt_pf.plot(
            title="Portfolio Backtest Results",
            subplots=["value", "cash", "drawdowns"],
        )
        if p is not None:
            plots.append(p)

        symbols = vbt_pf.orders.records_readable["Column"].unique().tolist()

        for tic in symbols:
            p = vbt_pf.plot(
                subplots=["asset_flow", "trade_pnl", "cum_returns", "trades"],
                title=f"{tic} Backtest Results",
                column=tic,
                group_by=False,
            )
            if p is not None:
                plots.append(p)

        return plots

    def plot(self):
        """
        Plot the results of the backtest.
        """
        for p in self._get_plots():
            p.show()

    def stats(self):
        """
        Return the statistics of the backtest.
        """
        return self.as_vbt_pf().stats()

    def orders(self):
        """
        Return the orders of the backtest.
        """
        return self.as_vbt_pf().orders.records_readable

    def trades(self):
        """
        Return the trades of the backtest.
        """
        return self.as_vbt_pf().trades.records_readable

    def __repr__(self) -> str:
        """
        String representation of the Portfolio.
        """
        return f"Portfolio(initial_cash={self.initial_cash}, total_value={self.total_value}, cash={self.cash}, nav={self.nav}, df_shape={self.df.shape})"

    def get_positions(self) -> defaultdict:
        """
        Return the current positions in the portfolio.
        """
        return self._positions

    def analysis(self, analysis_config):
        rprint(f"\nStats:\n{self.stats()}")

        if analysis_config.plot:
            self.plot()
        if analysis_config.save_plots:
            self.save_plots(ProjectPath.BACKTEST_DIR)
        if analysis_config.to_csv:
            self.get_positions().to_csv(ProjectPath.BACKTEST_DIR / "positions.csv")
            self.orders().to_csv(ProjectPath.BACKTEST_DIR / "orders.csv")
            self.trades().to_csv(ProjectPath.BACKTEST_DIR / "trades.csv")
