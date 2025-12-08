import logging
from abc import ABC

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient as AlpacaTradingClient

from trading.cli.trading.trade_config import BrokerType, RRTradeConfig
from trading.src.user_cache.user_cache import UserCache


class TradingClient(ABC):
    @classmethod
    def from_config(cls, config: RRTradeConfig, live: bool = False) -> "TradingClient":
        if config.broker == BrokerType.ALPACA:
            return AlpacaClient(live)
        elif config.broker == BrokerType.LOCAL:
            return LocalTradingClient()
        elif config.broker == BrokerType.REMOTE:
            return RemoteTradingClient()
        else:
            raise ValueError(f"Unsupported broker type: {config.broker}")

    def __init__(self, live: bool = False):
        self.live = live

    def get_account(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def execute_trades(self, actions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        raise NotImplementedError("This method should be implemented in a subclass.")

    def state(self, symbols: list[str]) -> np.ndarray:
        raise NotImplementedError("This method should be implemented in a subclass.")


class LocalTradingClient(TradingClient):
    def __init__(self):
        super().__init__(live=False)
        logging.info("Initialized Local Trading Client")

    def get_account(self):
        return {"balance": 1_000_000.0}

    def execute_trades(self, actions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        return actions, 0.0

    def state(self, symbols: list[str]) -> np.ndarray:
        return np.zeros(len(symbols))


class RemoteTradingClient(TradingClient):
    def __init__(self):
        super().__init__(live=True)
        logging.info("Initialized Remote Trading Client")

    def get_account(self):
        return {"balance": 1_000_000.0}

    def execute_trades(self, actions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        return actions, 0.0

    def state(self, symbols: list[str]) -> np.ndarray:
        return np.zeros(len(symbols))


class AlpacaClient(TradingClient):
    def __init__(self, live: bool = False):
        user_cache = UserCache().load()
        if live:
            self.alpaca_api_key = user_cache.alpaca_api_secret_live
            self.alpaca_api_secret = user_cache.alpaca_api_key_live
        elif not live:
            self.alpaca_api_key = user_cache.alpaca_api_key
            self.alpaca_api_secret = user_cache.alpaca_api_secret

        self.client: AlpacaTradingClient = AlpacaTradingClient(
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

    def get_clock(self):
        return self.client.get_clock()

    def get_calendar(self):
        return self.client.get_calendar()

    def execute_trades(self, actions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        logging.info("Executing trades via Alpaca")
        return actions, 0.0
