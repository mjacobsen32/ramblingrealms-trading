import datetime
import logging
from pyexpat import features
from typing import Any

import numpy as np
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrameUnit

from trading.cli.alg.config import DataConfig, DataRequests, FeatureConfig, StockEnv
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
    def __init__(
        self,
        config: RRTradeConfig,
        market_data_client: Any,
        alpaca_account_client: Any,
        live: bool,
        predict_time: datetime.datetime | None = None,
        end_predict_time: datetime.datetime | None = None,
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
        self.portfolio_config = self.env_config.portfolio_config
        self.config.portfolio_config = self.portfolio_config
        self.market_data_client = market_data_client

        feature_cfg = FeatureConfig.model_validate(self.meta_data)
        self.active_features = getattr(feature_cfg, "features", [])
        logging.debug("Active features: %s", self.active_features)

        self.live = live
        self.trading_client = TradingClient.from_config(
            config=self.config, alpaca_account_client=alpaca_account_client, live=live
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
            data=self._load_data(predict_time, end_predict_time).df,
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

    def _load_data(
        self,
        predict_time: datetime.datetime | None = None,
        end_predict_time: datetime.datetime | None = None,
    ) -> DataLoader:
        calendar = self.trading_client.alpaca_account_client.get_calendar()
        if predict_time is not None and end_predict_time is not None:
            range_of_days = end_predict_time.date() - predict_time.date()
        else:
            range_of_days = datetime.timedelta(days=1)
        min_window = (
            max(min_window_size(self.active_features), self.env_config.lookback_window)
            + range_of_days.days
            + 1
        )
        logging.info("Determined min window size from features: %d", min_window)
        # TODO this does not allow for live trading in intraday sessions
        if predict_time is None:
            prediction_time = datetime.datetime.now().date()
        else:
            prediction_time = predict_time.date()
            if end_predict_time is not None:
                end_prediction_time = end_predict_time.date()

        window = [
            entry
            for entry in calendar
            if entry.date
            < (prediction_time if end_predict_time is None else end_prediction_time)
        ][-(min_window):]
        start, end = window[0], window[-1]
        logging.info("Loading data for model trade: [%s, %s]", start.date, end.date)

        data_config = self.data_config
        data_config.start_date = str(start.date)
        data_config.end_date = str(end.date)
        data_config.cache_enabled = (
            False if self.live else data_config.cache_enabled
        )  # no caching for trading live

        logging.debug("Data Config: %s", data_config.model_dump_json(indent=4))

        feature_config = FeatureConfig(
            features=self.meta_data["features"], fill_strategy="interpolate"
        )

        return DataLoader(data_config=data_config, feature_config=feature_config)

    def get_prices(self) -> np.ndarray:
        req = StockLatestTradeRequest(symbol_or_symbols=self.active_symbols)
        latest = self.market_data_client.get_stock_latest_trade(req)
        prices = [trade.price for trade in latest.values()]
        return np.array(prices)

    def run_model(self, predict_time: datetime.datetime | None = None) -> None:

        obs, _ = self.env.reset()
        terminated, truncated = False, False

        while not terminated and not truncated:
            action, _states = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)

        return None
