import logging

import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest

from trading.cli.trading.trade_config import BrokerType, RRTradeConfig
from trading.src.alg.agents.agents import Agent
from trading.src.user_cache.user_cache import UserCache


class Trade:
    def __init__(self, config: RRTradeConfig, live: bool = False):
        self.config = config
        self.current_cash: float = 0.0
        self.current_positions: list[float] = []
        self.live = live

    @classmethod
    def from_config(cls, config: RRTradeConfig, live: bool = False) -> "Trade":
        if config.broker == BrokerType.ALPACA:
            return AlpacaTrade(config, live)
        else:
            raise ValueError(f"Unsupported broker type: {config.broker}")

    def initialize(self):
        pass

    def run_model(self):
        pass

    def _execute_trade(self, action: str):
        pass


class AlpacaTrade(Trade):
    def __init__(self, config: RRTradeConfig, live: bool = False):
        super().__init__(config, live)
        user_cache = UserCache().load()
        if live:
            self.alpaca_api_key = user_cache.alpaca_api_secret_live
            self.alpaca_api_secret = user_cache.alpaca_api_key_live
            self.BASE_URL = "https://api.alpaca.markets/v2"
        elif not live:
            self.alpaca_api_key = user_cache.alpaca_api_key
            self.alpaca_api_secret = user_cache.alpaca_api_secret
            self.BASE_URL = "https://paper-api.alpaca.markets/v2"
        self.trading_client = TradingClient(
            self.alpaca_api_key.get_secret_value(),
            self.alpaca_api_secret.get_secret_value(),
            paper=not live,
        )

    def initialize(self):
        super().initialize()
        account_details = self.trading_client.get_account()
        logging.info("Initialized account: %s", account_details.account_number)
        positions = self.trading_client.get_all_positions()
        self.model, meta_data = Agent.load_agent(self.config.model_path.as_path(), None)
        self.active_symbols = meta_data.get("symbols", [])
        self.active_features = meta_data.get("features", [])
        logging.info(
            "Loaded model type: '%s' version: '%s'\nSymbols: %s\nFeatures: %s",
            meta_data.get("type", "Unknown"),
            meta_data.get("version", "Unknown"),
            self.active_symbols,
            self.active_features,
        )
        logging.info("Positions: %s", positions)

    def run_model(self):
        super().run_model()

    def _execute_trade(self, action: str):
        super()._execute_trade(action)
