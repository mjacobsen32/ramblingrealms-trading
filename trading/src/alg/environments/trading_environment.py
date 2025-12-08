import logging
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit
from gymnasium import spaces

from trading.cli.alg.config import StockEnv, TradeMode
from trading.src.alg.environments.reward_functions.reward_function_factory import (
    reward_factory_method,
)
from trading.src.features import utils as feature_utils
from trading.src.features.generic_features import Feature
from trading.src.portfolio.portfolio import Portfolio, PositionManager


class TradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning agents.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cfg: StockEnv,
        features: List[Feature],
        time_step: tuple[TimeFrameUnit, int] = (TimeFrameUnit.Day, 1),
    ):
        self.symbols = data.index.get_level_values("symbol").unique().tolist()
        self.stock_dimension = len(data.index.get_level_values("symbol").unique())
        self.features = features
        self.feature_cols = feature_utils.get_feature_cols(features=self.features)
        self.init_data(data)
        self.cfg: StockEnv = cfg
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
        self.pf: Portfolio = Portfolio(
            cfg=cfg.portfolio_config,
            symbols=data.index.get_level_values("symbol").unique(),
            time_step=time_step,
            position_manager=position_manager,
        )
        self.reward_function = reward_factory_method(cfg.reward_config, self.pf.state())
        self.observation_index = self.cfg.lookback_window

        # Define action space: continuous [-1,1] per stock (sell, hold, buy)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.stock_dimension,),
            dtype=(
                np.float32
                if self.cfg.portfolio_config.trade_mode == TradeMode.CONTINUOUS
                else np.int32
            ),
        )

        # Environment state space
        state_space = (
            1  # cash balance
            + 2 * self.stock_dimension  # stock prices and shares held
            + (
                (len(self.feature_cols) * self.stock_dimension)
                * (self.cfg.lookback_window + 1)
            )  # technical indicators for each stock
        )
        logging.info(
            "State space: %s, Features (%s): %s, Stock Dimension: %s, action space: %s, lookback window: %i",
            state_space,
            len(self.feature_cols),
            self.feature_cols,
            self.stock_dimension,
            self.action_space.shape,
            self.cfg.lookback_window,
        )
        # State space (cash + owned shares + prices + indicators)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32
        )

        self._reset_internal_states()

    def init_data(self, data: pd.DataFrame):
        """
        Set the data for the environment.
        Args:
            data: DataFrame containing the trading data.
        """
        self.data = data.copy()
        self.data["size"] = 0.0
        self.data["profit"] = 0.0
        self.data["action"] = 0.0  # Initialize action column
        self.data["timestamp"] = self.data.index.get_level_values("timestamp")
        self.timestamps = data.index.get_level_values("timestamp").unique().to_list()
        self.max_steps = len(self.timestamps) - 1
        self.stats = pd.DataFrame(
            0.0,
            index=self.timestamps,
            columns=["cum_returns", "returns", "net_value"],
            dtype=np.float32,
        )

    def _reset_internal_states(self):
        self.observation_index = self.cfg.lookback_window
        self.terminal = False
        self.pf.reset()
        self.observation_timestamp = self.data.index.get_level_values(
            "timestamp"
        ).unique()
        logging.debug("Environment reset:\n%s", self.render())

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed, options=options)
        self._reset_internal_states()
        return self._get_observation(), {}

    @classmethod
    def observation(
        cls,
        df: pd.DataFrame,
        portfolio_state: np.ndarray,
        feature_cols: list,
        prices: np.ndarray,
    ) -> np.ndarray:
        indicators = df[feature_cols].to_numpy().flatten()
        logging.debug(
            "Portfolio state: %s\nPrices: %s\nIndicators: %s",
            portfolio_state.shape,
            prices.shape,
            indicators.shape,
        )
        c = np.concatenate([portfolio_state, prices, indicators], axis=0)
        logging.debug("Observation: %s", c)
        return c

    def _get_observation(self, i: int = -1) -> np.ndarray:
        """Get the current observation from the environment.

        Returns:
            np.ndarray: The current observation. [cash, positions, prices, indicators]
        """
        if i == -1:
            i = self.observation_index
        df = self.data.loc[
            self.observation_timestamp[
                i - self.cfg.lookback_window
            ] : self.observation_timestamp[i]
        ]
        prices = self.data.loc[[self.observation_timestamp[i]]]["price"].to_numpy()

        return self.observation(
            df, np.asarray(self.pf.state()), self.feature_cols, prices
        )

    def render(self):
        return "Day: {}\nSlice: {}\nTickers: {}, Observation Space: {}, Action Space: {}, Features: {}, Reward Function: {}, {}".format(
            self.observation_timestamp[self.observation_index],
            self.data.loc[self.observation_timestamp[self.observation_index]],
            self.data.index.get_level_values("symbol").unique().tolist(),
            self.observation_space.shape,
            self.action_space.shape,
            self.feature_cols,
            self.reward_function,
            self.pf,
        )

    def step(self, action):
        """
        Execute one time step within the environment.
        Args:
            action: The action to be taken by the agent, which is a vector of size equal to the number of stocks.
        Returns:
            observation: The new state of the environment after taking the action.
            reward: The reward received after taking the action.
            done: A boolean indicating whether the episode has ended.
            truncated: A boolean indicating whether the episode was truncated.
            info: Additional information about the environment.
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

        d = self.pf.step(df=date_slice, normalized_actions=True)

        self.stats.loc[
            self.observation_timestamp[self.observation_index], "net_value"
        ] = self.pf.net_value()
        previous_net_value = (
            self.stats.loc[
                self.observation_timestamp[self.observation_index - 1], "net_value"
            ]
            if self.observation_index > 0
            else self.pf.initial_cash
        )
        self.stats.loc[
            self.observation_timestamp[self.observation_index], "returns"
        ] = (self.pf.net_value() - previous_net_value) / previous_net_value

        self.stats.loc[
            self.observation_timestamp[self.observation_index], "cum_returns"
        ] = (self.pf.net_value() - self.pf.initial_cash) / self.pf.initial_cash

        ret_info = {
            "net_value": self.pf.net_value(),
            "profit_change": d["profit"],
        }

        logging.debug("Env State: %s", self)
        logging.debug("Portfolio State: %s", self.pf)
        logging.debug("Step Profit: %s", d["profit"])

        stats_slice = self.stats.iloc[0 : self.observation_index + 1]
        ret = (
            self._get_observation(),
            self.reward_function.compute_reward(
                pf=self.pf, df=stats_slice, realized_profit=d["profit"]
            ),
            self.terminal,
            False,
            ret_info,
        )
        self.observation_index += 1
        self.terminal = self.observation_index >= self.max_steps - 1
        return ret
