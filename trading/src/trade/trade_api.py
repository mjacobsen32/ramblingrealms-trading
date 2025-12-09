import datetime
import logging

import numpy as np
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrameUnit

from trading.cli.alg.config import DataConfig, DataRequests, FeatureConfig
from trading.cli.trading.trade_config import PortfolioConfig, RRTradeConfig
from trading.src.alg.agents.agents import Agent
from trading.src.alg.data_process.data_loader import DataLoader, DataSourceType
from trading.src.alg.environments.trading_environment import TradingEnv
from trading.src.features.utils import get_feature_cols, min_window_size
from trading.src.portfolio.portfolio import (
    Portfolio,
    TradeMode,
)
from trading.src.portfolio.position import LivePositionManager
from trading.src.trade.trade_clients import TradingClient
from trading.src.user_cache.user_cache import UserCache


class Trade:
    def __init__(self, config: RRTradeConfig, live: bool):
        self.config = config

        self.model, self.meta_data = Agent.load_agent(config.model_path.as_path(), None)
        self.active_symbols = self.meta_data.get("symbols", [])
        logging.info(
            "Meta Data loaded for model: %s:%s",
            self.meta_data["type"],
            self.meta_data["version"],
        )
        feature_cfg = FeatureConfig.model_validate(self.meta_data)
        self.active_features = getattr(feature_cfg, "features", [])
        logging.debug("Active features: %s", self.active_features)
        self.env_config = self.meta_data.get("env_config", {})
        self.portfolio_config = PortfolioConfig(
            **self.env_config.get("portfolio_config", {})
        )

        self.data_config = DataConfig.model_validate(self.meta_data["data_config"])

        self.live = live
        self.trading_client = TradingClient.from_config(config, live)

        user_cache = UserCache().load()
        alpaca_api_key = user_cache.alpaca_api_key
        alpaca_api_secret = user_cache.alpaca_api_secret

        self.market_data_client = StockHistoricalDataClient(
            alpaca_api_key.get_secret_value(), alpaca_api_secret.get_secret_value()
        )

        position_manager = LivePositionManager(
            trading_client=self.trading_client,
            symbols=self.active_symbols,
            max_lots=(
                None
                if self.portfolio_config.trade_mode == TradeMode.CONTINUOUS
                else self.portfolio_config.max_positions
            ),
            maintain_history=False,
            initial_cash=self.portfolio_config.initial_cash,
            initial_prices=self.get_prices(),
        )
        self.pf = Portfolio(
            self.portfolio_config,
            self.active_symbols,
            position_manager,
            (TimeFrameUnit.Day, 1),
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

    def _load_data(self, datetime_now: datetime.datetime | None = None) -> DataLoader:
        calendar = self.trading_client.alpaca_account_client.get_calendar()
        min_window = min_window_size(self.active_features) + 1
        logging.info("Determined min window size from features: %d", min_window)
        # TODO this does not allow for live trading in intraday sessions
        if datetime_now is None:
            prediction_time = datetime.datetime.now().date()
        else:
            prediction_time = datetime_now.date()

        window = [entry for entry in calendar if entry.date < prediction_time][
            -(min_window):
        ]
        start, end = window[0], window[-1]
        logging.info("Loading data for model trade: [%s, %s]", start.date, end.date)

        # TODO offload requests meta data construction
        requests = [
            DataRequests(
                dataset_name="live",
                source=DataSourceType.ALPACA,
                endpoint="StockBarsRequest",
                kwargs={
                    "symbol_or_symbols": self.active_symbols,
                    "adjustment": "split",
                },
            )
        ]

        data_config = self.data_config
        data_config.start_date = str(start.date)
        data_config.end_date = str(end.date)
        data_config.requests = requests

        logging.debug("Data Config: %s", data_config.model_dump_json(indent=4))

        feature_config = FeatureConfig(
            features=self.meta_data["features"], fill_strategy="interpolate"
        )
        return DataLoader(
            data_config=data_config, feature_config=feature_config, fetch_data=True
        )

    def get_prices(self) -> np.ndarray:
        req = StockLatestTradeRequest(symbol_or_symbols=self.active_symbols)
        latest = self.market_data_client.get_stock_latest_trade(req)
        prices = [trade.price for trade in latest.values()]
        return np.array(prices)

    def run_model(self, predict_time: datetime.datetime | None = None) -> dict:
        self.data_loader = self._load_data(predict_time)
        df = self.data_loader.df
        prices = self.get_prices()

        logging.debug("Latest prices: %s", prices)
        logging.debug("df head:\n%s", self.data_loader.df.tail())
        logging.debug("portfolio_state:\n%s", self.pf.state())
        logging.debug("feature columns: %s", get_feature_cols(self.active_features))

        obs = TradingEnv.observation(
            df[-(self.env_config.get("lookback_window") + 1) :],
            self.pf.state(),
            get_feature_cols(self.active_features),
            prices,
        )

        logging.debug("Observation: %s", obs)
        logging.debug("Observation shape: %s", obs.shape)

        actions, _states = self.model.predict(obs)

        logging.info("Model Actions: %s", actions)
        logging.debug("data: %s", df.tail())

        df["size"] = 1.0
        df["profit"] = 0.0
        df["price"] = 1.0
        df["action"] = 1.0
        df["timestamp"] = df.index.get_level_values("timestamp")

        current_slice = df.iloc[[-1]]
        current_slice = current_slice.droplevel(0)
        current_slice.loc[:, "action"] = actions
        current_slice.loc[:, "price"] = prices

        ret = self.pf.step(current_slice, True)

        logging.info("Scaled actions: %s", ret["scaled_actions"])
        logging.info("Profit: %s", ret["profit"])

        return ret
