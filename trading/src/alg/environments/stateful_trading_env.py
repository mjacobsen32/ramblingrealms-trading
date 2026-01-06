import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit

from trading.cli.alg.config import StockEnv
from trading.src.alg.environments.base_environment import BaseTradingEnv
from trading.src.alg.environments.reward_functions.reward_function_factory import (
    reward_factory_method,
)
from trading.src.features.generic_features import Feature
from trading.src.portfolio.portfolio import Portfolio, PositionManager


class StatefulTradingEnv(BaseTradingEnv):
    """
    Stateful trading environment for backtesting and paper trading.
    Maintains full portfolio state, position tracking, and enforces trading constraints.
    This is the environment that should be used for realistic testing and evaluation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cfg: StockEnv,
        features: List[Feature],
        time_step: tuple[TimeFrameUnit, int] = (TimeFrameUnit.Day, 1),
        position_manager: PositionManager | None = None,
    ):
        super().__init__(data, cfg, features, time_step)

        # Initialize full portfolio management
        from trading.cli.alg.config import TradeMode

        if position_manager is None:
            position_manager = PositionManager(
                symbols=self.symbols,
                max_lots=(
                    None
                    if cfg.portfolio_config.trade_mode == TradeMode.CONTINUOUS
                    else cfg.portfolio_config.max_positions
                ),
                maintain_history=cfg.portfolio_config.maintain_history,
                initial_cash=cfg.portfolio_config.initial_cash,
            )
        elif isinstance(position_manager, PositionManager):
            pass

        self.pf: Portfolio = Portfolio(
            cfg=cfg.portfolio_config,
            symbols=data.index.get_level_values("symbol").unique(),
            time_step=time_step,
            position_manager=position_manager,
        )

        # Reward function for evaluation
        self.reward_function = reward_factory_method(cfg.reward_config, self.pf.state())

        # Statistics tracking for reward calculations
        self.stats = pd.DataFrame(
            0.0,
            index=self.timestamps,
            columns=["cum_returns", "returns", "net_value"],
            dtype=np.float32,
        )

        # Initialize action tracking in data
        self.data["size"] = 0.0
        self.data["profit"] = 0.0
        self.data["action"] = 0.0

        logging.info("StatefulTradingEnv initialized with full portfolio tracking")

    def _get_observation(self, i: int = -1) -> np.ndarray:
        """Get the current observation with full portfolio state."""
        if i == -1:
            i = self.observation_index

        df = self._get_observation_df(i)
        prices = self._get_prices(i)

        return self.observation(
            df, np.asarray(self.pf.state()), self.feature_cols, prices
        )

    def _reset_internal_states(self, timestamp: pd.Timestamp | None = None):
        """Reset internal states including portfolio."""
        super()._reset_internal_states(timestamp=timestamp)
        # Initialize action tracking in data
        self.data["size"] = 0.0
        self.data["profit"] = 0.0
        self.data["action"] = 0.0
        self.pf.reset()

    def reset(
        self,
        *,
        timestamp: pd.Timestamp | None = None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset the environment to its initial state."""
        super().reset(seed=seed, options=options)
        self._reset_internal_states(timestamp=timestamp)
        return self._get_observation(), {}

    def render(self):
        """Render environment state with portfolio information."""
        base_render = super().render()
        return f"{base_render}, Reward Function: {self.reward_function}, {self.pf}"

    def step(self, action):
        """
        Execute one time step with full portfolio management.
        This includes position tracking, trade enforcement, and proper reward calculation.
        """
        if self.terminal:
            return (
                self._get_observation(self.observation_index - 1),
                0,
                self.terminal,
                False,
                {},
            )

        date_slice = self.data.loc[self.observation_timestamp[self.observation_index]]
        date_slice.loc[:, "action"] = action

        logging.debug("action: %s, date_slice: %s", action, date_slice)

        # Execute trades through portfolio manager (enforces constraints)
        d = self.pf.step(df=date_slice, normalized_actions=True)

        # Update statistics
        self.stats.loc[
            self.observation_timestamp[self.observation_index], "net_value"
        ] = self.pf.position_manager.net_value()

        previous_net_value = (
            self.stats.loc[
                self.observation_timestamp[self.observation_index - 1], "net_value"
            ]
            if self.observation_index > self.cfg.lookback_window
            else self.pf.position_manager.initial_cash()
        )

        self.stats.loc[
            self.observation_timestamp[self.observation_index], "returns"
        ] = (
            self.pf.position_manager.net_value() - previous_net_value
        ) / previous_net_value

        self.stats.loc[
            self.observation_timestamp[self.observation_index], "cum_returns"
        ] = (
            self.pf.position_manager.net_value()
            - self.pf.position_manager.initial_cash()
        ) / self.pf.position_manager.initial_cash()

        ret_info = {
            "timestamp": self.observation_timestamp[self.observation_index],
            "net_value": self.pf.position_manager.net_value(),
            "profit_change": d["profit"],
            "orders": d["orders"],
        }

        logging.debug("Env State: %s", self)
        logging.debug("Portfolio State: %s", self.pf)
        logging.debug("Step Profit: %s", d["profit"])

        # Calculate reward using configured reward function
        stats_slice = self.stats.iloc[0 : self.observation_index + 1]
        reward = self.reward_function.compute_reward(
            pf=self.pf, df=stats_slice, realized_profit=d["profit"]
        )

        # Move to next step
        self.observation_index += 1
        self.terminal = self.observation_index >= self.max_steps
        return (
            self._get_observation() if not self.terminal else None,
            reward,
            self.terminal,
            False,
            ret_info,
        )
