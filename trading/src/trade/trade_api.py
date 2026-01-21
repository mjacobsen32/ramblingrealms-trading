import datetime
import logging
from pyexpat import features
from typing import Any

import numpy as np
import pandas as pd
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrameUnit
from alpaca.trading.requests import GetCalendarRequest

from trading.cli.alg.config import (
    AgentConfig,
    DataConfig,
    FeatureConfig,
    StockEnv,
)
from trading.cli.trading.trade_config import RRTradeConfig
from trading.src.alg.agents.agents import Agent
from trading.src.alg.data_process.data_loader import DataLoader, DataSourceType
from trading.src.alg.environments.stateful_trading_env import StatefulTradingEnv
from trading.src.alg.environments.trading_environment import TradingEnv
from trading.src.features.utils import get_feature_cols, min_window_size
from trading.src.portfolio.portfolio import (
    Portfolio,
    TradeMode,
)
from trading.src.portfolio.position import LivePositionManager
from trading.src.trade.trade_clients import TradingClient


class Trade:

    class LiveTradeError(Exception):
        """Custom exception for live trading errors."""

        OUT_OF_RANGE = "OUT_OF_RANGE"

        def __init__(self, error_type: str, message: str):
            self.error_type = error_type
            self.message = message
            super().__init__(f"[{error_type}] {message}")

    def __init__(
        self,
        config: RRTradeConfig,
        market_data_client: Any,
        alpaca_account_client: Any,
        live: bool,
        predict_time: datetime.datetime,
        end_predict_time: datetime.datetime,
        fetch_data: bool = True,
    ):
        self.config = config
        self.model, self.meta_data = Agent.load_agent(config.model_path.as_path(), None)
        self.active_symbols = self.meta_data.get("symbols", [])
        logging.info(
            "Meta Data loaded for model: %s:%s",
            self.meta_data["type"],
            self.meta_data["version"],
        )
        self.env_config = StockEnv.model_validate(self.meta_data["env_config"])
        self.data_config = DataConfig.model_validate(self.meta_data["data_config"])
        self.agent_config = AgentConfig.model_validate(self.meta_data["agent_config"])
        self.portfolio_config = self.env_config.portfolio_config
        self.config.portfolio_config = self.portfolio_config
        self.market_data_client = market_data_client
        self.alpaca_account_client = alpaca_account_client

        self.feature_cfg = FeatureConfig.model_validate(self.meta_data)
        self.active_features = getattr(self.feature_cfg, "features", [])
        logging.debug("Active features: %s", self.active_features)

        self.live = live
        self.trading_client = TradingClient.from_config(
            config=self.config,
            alpaca_account_client=self.alpaca_account_client,
            live=live,
        )

        position_manager = LivePositionManager(
            trading_client=self.trading_client,
            symbols=self.active_symbols,
            max_lots=(
                None
                if self.portfolio_config.trade_mode == TradeMode.CONTINUOUS
                else self.portfolio_config.max_positions
            ),
            maintain_history=True,
            initial_cash=self.portfolio_config.initial_cash,
            initial_prices=self.get_prices(),
        )

        self.env = StatefulTradingEnv(
            data=self._load_data(
                predict_time, end_predict_time, fetch_data=fetch_data
            ).df,
            cfg=self.env_config,
            features=self.active_features,
            time_step=(
                self.data_config.time_step_unit,
                self.data_config.time_step_period,
            ),
            position_manager=position_manager,
        )

        logging.debug(
            "Loaded model type: '%s' version: '%s'\nPortfolio Config: %s\nEnv Config: %s\nSymbols: %s\nFeatures: %s",
            self.meta_data.get("type", "Unknown"),
            self.meta_data.get("version", "Unknown"),
            self.portfolio_config,
            self.env_config,
            self.active_symbols,
            self.active_features,
        )

    def __del__(self):

        agent_config_minus_sensitive = self.agent_config.model_dump()
        if "save_path" in agent_config_minus_sensitive:
            agent_config_minus_sensitive.pop("save_path")
        self.trading_client.write_meta_data(
            {
                "type": self.meta_data.get("type", "Unknown"),
                "version": self.meta_data.get("version", "Unknown"),
                "active": self.config.active,
                "id": self.config.id,
                "asset_exchanges": self.config.asset_exchanges,
                "symbols": self.active_symbols,
                "features": self.active_features,
                "env_config": self.env_config.model_dump(),
                "agent_config": agent_config_minus_sensitive,
            }
        )

    def _load_data(
        self,
        predict_time: datetime.datetime,
        end_predict_time: datetime.datetime,
        fetch_data: bool = True,
    ) -> DataLoader:
        calendar = self.trading_client.alpaca_account_client.get_calendar()
        range_of_days = (end_predict_time.date() - predict_time.date()).days

        effective_lookback = (
            self.env_config.lookback_window
            + min_window_size(self.active_features)
            + range_of_days
            + 1
        )

        # Filter and access directly with negative indexing
        filtered_calendar = [
            entry for entry in calendar if entry.date <= end_predict_time.date()
        ]
        data_start = filtered_calendar[-effective_lookback]

        logging.info(
            "Loading data window: [start_predict: %s, end_predict: %s]\n\tlookback_window: %s, data_start: %s",
            predict_time,
            end_predict_time,
            effective_lookback,
            data_start,
        )

        # Ensure timezone awareness, default to UTC
        start_dt = datetime.datetime.combine(
            data_start.date, datetime.time.min, tzinfo=datetime.timezone.utc
        )
        end_dt = (
            end_predict_time
            if end_predict_time.tzinfo is not None
            else end_predict_time.replace(tzinfo=datetime.timezone.utc)
        )

        data_config = self.data_config
        data_config.start_date = start_dt.isoformat()
        data_config.end_date = end_dt.isoformat()
        data_config.cache_enabled = not self.live and data_config.cache_enabled

        logging.debug("Data Config: %s", data_config.model_dump_json(indent=4))

        feature_config = FeatureConfig(
            features=self.meta_data["features"], fill_strategy="interpolate"
        )

        return DataLoader(
            data_config=data_config,
            feature_config=feature_config,
            fetch_data=fetch_data,
            alpaca_history_client=self.market_data_client,
            alpaca_trading_client=self.alpaca_account_client,
        )

    def get_prices(self) -> np.ndarray:
        req = StockLatestTradeRequest(symbol_or_symbols=self.active_symbols)
        latest = self.market_data_client.get_stock_latest_trade(req)
        prices = [trade.price for trade in latest.values()]
        return np.array(prices)

    def range_includes_open_markets(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> bool:
        request = GetCalendarRequest(start=start.date(), end=end.date())
        calendar = self.trading_client.alpaca_account_client.get_calendar(request)
        if len(calendar) == 0:
            logging.debug("No market days in the given range [%s - %s]", start, end)
            return False
        return True

    def run_model(
        self,
        predict_time: datetime.datetime,
        end_predict_time: datetime.datetime,
    ) -> list[dict[str, Any]]:

        if not self.range_includes_open_markets(predict_time, end_predict_time):
            raise Trade.LiveTradeError(
                Trade.LiveTradeError.OUT_OF_RANGE,
                f"Predict time range [{predict_time} - {end_predict_time}] does not include any open market hours.",
            )
        if (
            predict_time
            < self.env.data.index.get_level_values("timestamp").min().to_pydatetime()
        ):
            raise Trade.LiveTradeError(
                Trade.LiveTradeError.OUT_OF_RANGE,
                f"Predict time {predict_time} is before available data starting at {self.env.data.index.get_level_values('timestamp').min()}.",
            )
        if (
            end_predict_time
            > self.env.data.index.get_level_values("timestamp").max().to_pydatetime()
        ):
            raise Trade.LiveTradeError(
                Trade.LiveTradeError.OUT_OF_RANGE,
                f"End predict time {end_predict_time} is after available data ending at {self.env.data.index.get_level_values('timestamp').max()}.",
            )

        obs, _ = self.env.reset(timestamp=pd.Timestamp(predict_time))
        terminated, truncated = False, False
        ret: list[dict[str, Any]] = []

        end_time = pd.Timestamp(end_predict_time)

        # Check that observation data is available before logging or entering loop
        if self.env.observation_timestamp is None or self.env.observation_index is None:
            logging.warning("No observation data available")
            return ret

        logging.info(
            "Observation Index : [%s, %s]",
            self.env.observation_timestamp[self.env.observation_index],
            self.env.observation_timestamp[-1],
        )

        while (
            not terminated
            and not truncated
            and self.env.observation_timestamp is not None
            and self.env.observation_index is not None
            and end_time >= self.env.observation_timestamp[self.env.observation_index]
        ):
            action, _states = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            ret.append(info)

        return ret
