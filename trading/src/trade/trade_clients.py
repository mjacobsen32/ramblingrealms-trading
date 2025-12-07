import logging

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient as AlpacaTradingClient

from trading.cli.trading.trade_config import BrokerType, RRTradeConfig
from trading.src.user_cache.user_cache import UserCache


class TradingClient:
    @classmethod
    def from_config(cls, config: RRTradeConfig, live: bool = False) -> "TradingClient":
        if config.broker == BrokerType.ALPACA:
            return AlpacaClient(live)
        else:
            raise ValueError(f"Unsupported broker type: {config.broker}")

    def __init__(self, live: bool = False):
        self.live = live

    def get_account(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def execute_trades(self, actions: pd.DataFrame):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def state(self, symbols: list[str]) -> np.ndarray:
        raise NotImplementedError("This method should be implemented in a subclass.")


class AlpacaClient(TradingClient):
    def __init__(self, live: bool = False):
        user_cache = UserCache().load()
        if live:
            self.alpaca_api_key = user_cache.alpaca_api_secret_live
            self.alpaca_api_secret = user_cache.alpaca_api_key_live
            self.BASE_URL = "https://api.alpaca.markets/v2"
        elif not live:
            self.alpaca_api_key = user_cache.alpaca_api_key
            self.alpaca_api_secret = user_cache.alpaca_api_secret
            self.BASE_URL = "https://paper-api.alpaca.markets/v2"

        self.client = AlpacaTradingClient(
            self.alpaca_api_key.get_secret_value(),
            self.alpaca_api_secret.get_secret_value(),
            paper=not live,
        )

        self.account_details = self.client.get_account()
        logging.info(
            "Initialized Alpaca Client: %s", self.account_details.account_number
        )

    def get_account(self):
        return self.account_details

    def execute_trades(self, actions: pd.DataFrame):
        pass

    def state(self, symbols: list[str]) -> np.ndarray:
        positions_dict = {
            p.symbol: float(p.qty) for p in self.client.get_all_positions()
        }
        return np.array([positions_dict.get(symbol, 0.0) for symbol in symbols])
