import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from alpaca.data.timeframe import TimeFrameUnit
from plotly import io as pio
from vectorbt import _typing as tp

from trading.cli.alg.config import PortfolioConfig, ProjectPath, TradeMode
from trading.src.alg.portfolio.position import Position, PositionManager
from trading.src.utility.utils import time_frame_unit_to_pd_timedelta


class Portfolio:
    """
    Portfolio class for managing trading positions and cash flow.
    Very much Stateful
    """

    def __init__(
        self,
        cfg: PortfolioConfig,
        symbols: list[str],
        time_step: tuple[TimeFrameUnit, int] = (TimeFrameUnit.Day, 1),
    ):
        """
        Initialize the Portfolio
        """
        self.cfg = cfg
        self.initial_cash = cfg.initial_cash
        self.total_value: float = cfg.initial_cash
        self.cash = cfg.initial_cash
        self.nav: float = 0.0
        self.vbt_pf: vbt.Portfolio | None = None
        self.symbols = symbols
        self._positions = PositionManager(
            symbols=symbols,
            max_lots=(
                None if cfg.trade_mode == TradeMode.CONTINUOUS else cfg.max_positions
            ),
        )

        self.time_step = time_frame_unit_to_pd_timedelta(time_step)
        self.df: pd.DataFrame | None = None

    def as_vbt_pf(self, df: pd.DataFrame | None = None) -> vbt.Portfolio:
        """
        Update the portfolio with new data.
        """
        if self.vbt_pf is not None:
            return self.vbt_pf
        elif df is not None:
            self.df = df
        elif self.df is None:
            raise ValueError("No DataFrame provided")

        self.df.reset_index(level="symbol", inplace=True)
        self.df.set_index(["timestamp"], inplace=True)
        price = self.df.pivot(columns="symbol", values="price")
        close = self.df.pivot(columns="symbol", values="close")
        size = self.df.pivot(columns="symbol", values="size")

        self.vbt_pf = vbt.Portfolio.from_orders(
            close=close,
            price=price,
            size=size,
            size_type=0,  # amount
            init_cash=self.initial_cash,
            log=True,
            cash_sharing=True,
            freq="d",
        )
        return self.vbt_pf

    def state(self) -> np.ndarray:
        """
        Get the current state of the portfolio.
        Returns:
            pd.Series: [internal_cash, positions].
        """
        return np.concatenate([[self.cash], self._positions["holdings"]])

    def net_value(self) -> float:
        """
        Calculate the net value of the portfolio.
        """
        return self.total_value

    def enforce_trade_rules(
        self,
        df: pd.DataFrame,
        prices: np.ndarray,
        trade_mode: TradeMode = TradeMode.CONTINUOUS,
    ) -> np.ndarray:
        """
        Enforce trade rules on the actions.
        From trade sizes to actual trade sizes based on configuration and current portfolio state.
        @TODO clean up and make more efficient
        """

        buy_mask = df["size"] > 0
        sell_mask = df["size"] < 0
        """
            No short selling / no negative positions allowed (yet)
        """

        positions = self._positions["holdings"]
        if trade_mode == TradeMode.CONTINUOUS:
            # Sell proportionally based on signal strength
            df.loc[sell_mask, "size"] = np.clip(
                df.loc[sell_mask, "size"], -positions[sell_mask], 0
            )
        elif trade_mode == TradeMode.DISCRETE:
            # Sell entire position if sell action is triggered
            df.loc[sell_mask, "size"] = -positions[sell_mask]

        """
            Scale the actions to the portfolio value
        """

        buy_limit = self.cfg.trade_limit_percent * self.total_value

        buy_values = df.loc[buy_mask, "size"] * prices[buy_mask]

        # Cap each trade at buy_limit
        capped_values = np.minimum(buy_values, buy_limit)

        attempted_buy = capped_values.sum()

        if attempted_buy > self.cash:
            buy_limit = min(buy_limit, self.cash // len(buy_mask))

        logging.debug("Buy Limit dollar amount: %s", buy_limit)

        # Calculate the maximum shares allowed by hmax and buy_limit for each buy
        max_shares_hmax = self.cfg.hmax // prices[buy_mask]
        max_shares_buy_limit = buy_limit // prices[buy_mask]
        max_shares = np.minimum(max_shares_hmax, max_shares_buy_limit)

        # Clip the size to not exceed the minimum of both limits
        df.loc[buy_mask, "size"] = np.clip(df.loc[buy_mask, "size"], 0, max_shares)
        df.loc[~buy_mask & ~sell_mask, "size"] = 0.0
        logging.debug("Scaled Sizes: %s", df["size"])

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
        @TODO clean up and make more efficient
        """
        logging.debug("Raw Action: %s", df["action"])
        # Find actions above threshold
        above_thresh = df["action"].abs() > self.cfg.action_threshold

        # Calculate max shares allowed by hmax and trade_limit
        max_trade_value = min(
            self.cfg.hmax, self.cfg.trade_limit_percent * self.net_value()
        )
        max_shares = max_trade_value // prices

        # Vectorized, faster assignment using numpy
        size = np.zeros_like(df["action"].values, dtype=np.float64)
        actions = df["action"].values
        above_idx = np.where(above_thresh)[0]

        if trade_mode == TradeMode.DISCRETE:
            size[above_idx] = actions[above_idx].astype(np.float64) * max_shares[
                above_idx
            ].astype(np.float64)
        elif trade_mode == TradeMode.CONTINUOUS:
            size[above_idx] = np.round(
                actions[above_idx] * max_shares[above_thresh]
            ).astype(np.float64)

        df.loc[:, "size"] = size
        return size

    def update_position_batch(self, df: pd.DataFrame) -> float:
        """
        Update positions for a batch of tickers at a given timestamp.
        @TODO clean up and make more efficient
        """
        # Reduce df to a single datetime index (symbols only)
        df, step_profit = self._positions.step(df)

        self.cash = self.cash - (df["size"] * df["price"]).sum()
        self.nav = self._positions.nav(df["price"])
        self.total_value = self.cash + self.nav

        logging.debug("df: %s\nstep_profit: %s\n", df, step_profit)

        return step_profit

    def step(self, df: pd.DataFrame, normalized_actions: bool = False) -> dict:
        """
        Scale the actions and take a step in the portfolio environment.
        1. scale actions to trade size
        2. enforce trade limits
        3. update positions
        @TODO clean up and make more efficient
        """
        if normalized_actions:
            df.loc[:, "size"] = self.scale_actions(
                df, df["price"].values, self.cfg.trade_mode
            )

        df.loc[:, "size"] = self.enforce_trade_rules(
            df, df["price"].values, self.cfg.trade_mode
        )
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
        logging.info("Loaded backtest results from %s", file_path)
        ret = cls(cfg=cfg, symbols=[])
        ret.vbt_pf = pf
        return ret

    def save(self, file_path: str, df: pd.DataFrame | None = None):
        logging.info("Saving backtest results to %s", file_path)
        self.as_vbt_pf(df=df).save(file_path)

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
            # use pio.write_images when kaleido is upgraded to 1.x.x
            pio.write_image(p, path, format="svg", scale=2, width=1200, height=3600)

    def reset(self):
        """
        Reset the portfolio to an empty state.
        """
        self.initial_cash = self.initial_cash
        self.total_value = self.initial_cash
        self.cash = self.initial_cash
        self.nav = 0.0
        self.vbt_pf = None
        self._positions.reset()
        logging.debug("Portfolio has been reset.\n%s", self)

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
        """
         plots: [
             "value", 
             "cash", 
             "drawdowns", 
             "orders", 
             "trades", 
             "trade_pnl", 
             "asset_flow", 
             "cash_flow", 
             "assets", 
             "asset_value", 
             "cum_returns", 
             "underwater", 
             "gross_exposure", 
             "net_exposure"
            ]
        """
        plots: list[tp.BaseFigure] = []
        p = vbt_pf.plot(
            title="Portfolio Backtest Results",
            subplots=[
                "value",
                "cash",
                "drawdowns",
                "cash_flow",
                "asset_value",
                "cum_returns",
                "underwater",
                "gross_exposure",
                "net_exposure",
            ],
        )
        if p is not None:
            plots.append(p)

        symbols = vbt_pf.orders.records_readable["Column"].unique().tolist()

        for tic in symbols:
            p = vbt_pf.plot(
                subplots=[
                    "value",
                    "cash",
                    "drawdowns",
                    "orders",
                    "trades",
                    "trade_pnl",
                    "asset_flow",
                    "cash_flow",
                    "assets",
                    "asset_value",
                    "cum_returns",
                    "underwater",
                    "gross_exposure",
                    "net_exposure",
                ],
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
        pio.renderers.default = "browser"
        logging.info(
            "Generating portfolio plots...\nRendering with: %s", pio.renderers.default
        )
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
        return f"Portfolio(initial_cash={self.initial_cash}, total_value={self.total_value}, cash={self.cash}, nav={self.nav})"

    def get_positions(self) -> PositionManager:
        """
        Return the current positions in the portfolio.
        """
        return self._positions

    def analysis(self, analysis_config, df: pd.DataFrame | None = None):
        logging.info(f"\nStats:\n{self.stats()}")
        self.as_vbt_pf(df=df)

        bt_dir = (
            ProjectPath.BACKTEST_DIR
            if ProjectPath.BACKTEST_DIR is not None
            else Path.cwd()
        )

        if analysis_config.render_plots:
            self.plot()
        if analysis_config.save_plots:
            self.save_plots(bt_dir)
        if analysis_config.to_csv:
            self.get_positions().to_csv(str(bt_dir / "positions.csv"))
            self.orders().to_csv(bt_dir / "orders.csv")
            self.trades().to_csv(bt_dir / "trades.csv")
