import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient as AlpacaTradingClient
from alpaca.trading.enums import AccountStatus
from alpaca.trading.models import Position as AlpacaPosition
from alpaca.trading.models import TradeAccount
from alpaca.trading.requests import MarketOrderRequest

try:
    import boto3
except Exception:  # pragma: no cover - optional dependency for remote client
    boto3 = None

from trading.cli.trading.trade_config import BrokerType, RRTradeConfig
from trading.src.portfolio.position import Position, PositionDecoder, PositionEncoder
from trading.src.user_cache.user_cache import UserCache


def _json_default(obj: object):  # -> str | Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, deque):
        return list(obj)
    if isinstance(obj, Position):
        return PositionEncoder().default(obj)
    # Fallback to str
    return str(obj)


class TradingClient(ABC):
    @classmethod
    def from_config(cls, config: RRTradeConfig, live: bool = False) -> "TradingClient":
        if config.broker == BrokerType.ALPACA:
            return AlpacaClient(config=config, live=live)
        elif config.broker == BrokerType.LOCAL:
            return LocalTradingClient(config=config)
        elif config.broker == BrokerType.REMOTE:
            return RemoteTradingClient(config=config)
        else:
            raise ValueError(f"Unsupported broker type: {config.broker}")

    def __init__(self, live: bool, config: RRTradeConfig):
        user_cache = UserCache().load()

        self.live = live
        self.config = config
        self.alpaca_api_key = user_cache.alpaca_api_key
        self.alpaca_api_secret = user_cache.alpaca_api_secret

        self.alpaca_account_client: AlpacaTradingClient = AlpacaTradingClient(
            self.alpaca_api_key.get_secret_value(),
            self.alpaca_api_secret.get_secret_value(),
            paper=not live,
        )

        self._account = self._load_account()
        self._positions = self._load_positions()

    def get_clock(self):
        return self.alpaca_account_client.get_clock()

    def get_calendar(self):
        return self.alpaca_account_client.get_calendar()

    @property
    def account(self) -> TradeAccount:
        return self._account

    @property
    def positions(self) -> dict[str, deque[Position]]:
        return self._positions

    @abstractmethod
    def _load_positions(self) -> dict[str, deque[Position]]:
        raise NotImplementedError

    @abstractmethod
    def _load_account(self) -> TradeAccount:
        raise NotImplementedError

    @abstractmethod
    def execute_trades(
        self, actions: pd.DataFrame, positions: dict[str, deque[Position]]
    ) -> tuple[pd.DataFrame, float]:
        raise NotImplementedError


class LocalTradingClient(TradingClient):
    def __init__(self, config: RRTradeConfig):
        self.positions_path = self._resolve_path(config.positions_path, "positions")
        self.account_path = self._resolve_path(config.account_path, "account")
        super().__init__(live=False, config=config)
        logging.info("Initialized Local Trading Client")

    def __del__(self) -> None:
        self.account_path.parent.mkdir(parents=True, exist_ok=True)
        self.account_path.write_text(self.account.model_dump_json(indent=2))

    @staticmethod
    def _resolve_path(path_field, default_name: str) -> Path:
        if path_field is not None:
            return path_field.as_path()
        raise ValueError(f"No path configured for local trading {default_name} data")

    def _load_positions(self) -> dict[str, deque[Position]]:
        ret = dict[str, deque[Position]]()
        count = 0
        if not self.positions_path.exists():
            logging.info("No positions file found at %s", self.positions_path)
            return ret
        data = json.loads(self.positions_path.read_text())
        for key, positions in data.items():
            for pos in positions:
                count += 1
                ret.setdefault(key, deque()).append(
                    json.loads(json.dumps(pos), object_hook=PositionDecoder)
                )
        logging.info("Loaded %d positions from %s", count, self.positions_path)
        return ret

    def _load_account(self) -> TradeAccount:
        if self.config.portfolio_config is None:
            raise ValueError(
                "Portfolio config is required for local account initialization"
            )
        else:
            init_cash = self.config.portfolio_config.initial_cash
        if not self.account_path.exists():
            account = TradeAccount(
                id=self.config.id,
                account_number=self.config.account_number,
                status=AccountStatus.ACTIVE,
                cash=str(init_cash),
            )
        else:
            account = TradeAccount.model_validate_json(self.account_path.read_text())

        logging.info("Loaded account id: %s; cash: %s", account.id, account.cash)
        return account

    def execute_trades(
        self, actions: pd.DataFrame, positions: dict[str, deque[Position]]
    ) -> tuple[pd.DataFrame, float]:
        for sym, row in actions.iterrows():
            logging.info(
                "sym: %s, size: %s, at price: %s, with signal: %s",
                sym,
                row.get("size"),
                row.get("price"),
                row.get("action"),
            )

        self._positions = positions

        return actions, actions.get("profit", 0.0).sum()


class RemoteTradingClient(TradingClient):
    def __init__(self, config: RRTradeConfig):
        super().__init__(live=True, config=config)
        logging.info("Initialized Remote Trading Client")
        if boto3 is None:
            raise ImportError("boto3 is required for RemoteTradingClient")
        self.bucket = config.remote_bucket or config.broker_kwargs.get("remote_bucket")
        self.prefix = config.remote_prefix or config.broker_kwargs.get(
            "remote_prefix", ""
        )
        if not self.bucket:
            raise ValueError("RemoteTradingClient requires remote_bucket to be set")
        cache = UserCache().load()
        session_kwargs = {
            "aws_access_key_id": cache.r2_access_key_id.get_secret_value(),
            "aws_secret_access_key": cache.r2_secret_access_key.get_secret_value(),
        }
        if cache.r2_endpoint_url:
            session_kwargs["endpoint_url"] = cache.r2_endpoint_url
        self._s3 = boto3.client("s3", **session_kwargs)
        self.positions_key = config.broker_kwargs.get(
            "remote_positions_key", "positions.json"
        )
        self.account_key = config.broker_kwargs.get(
            "remote_account_key", "account.json"
        )

    def _load_positions(self) -> dict[str, deque[Position]]:
        return {}

    def _load_account(self) -> TradeAccount:
        return TradeAccount(
            id=UUID("00000000-0000-0000-0000-000000000000"),
            account_number="remote",
            status=AccountStatus.ACTIVE,
        )

    def execute_trades(
        self, actions: pd.DataFrame, positions: dict[str, deque[Position]]
    ) -> tuple[pd.DataFrame, float]:
        return actions, 0.0


class AlpacaClient(TradingClient):
    def __init__(self, config: RRTradeConfig, live: bool = False):
        user_cache = UserCache().load()

        if live:
            self.alpaca_api_key = user_cache.alpaca_api_key_live
            self.alpaca_api_secret = user_cache.alpaca_api_secret_live
        else:
            self.alpaca_api_key = user_cache.alpaca_api_key
            self.alpaca_api_secret = user_cache.alpaca_api_secret

        self.client: AlpacaTradingClient = AlpacaTradingClient(
            self.alpaca_api_key.get_secret_value(),
            self.alpaca_api_secret.get_secret_value(),
            paper=not live,
        )

        super().__init__(live=live, config=config)

    def _load_positions(self) -> dict[str, deque[Position]]:
        ret = dict[str, deque[Position]]()
        alpaca_positions = self.client.get_all_positions()
        for alpaca_pos in alpaca_positions:
            if isinstance(alpaca_pos, AlpacaPosition):
                pos = Position.from_alpaca_position(alpaca_pos)
            else:
                raise ValueError("Failed to load Alpaca position as Position")
            ret.setdefault(pos.symbol, deque()).append(pos)
        logging.info("Loaded %d positions from Alpaca", len(alpaca_positions))
        return ret

    def _load_account(self) -> TradeAccount:
        account = self.client.get_account()
        if isinstance(account, TradeAccount):
            logging.info(
                "Loaded Alpaca account id: %s; cash: %s", account.id, account.cash
            )
            return account
        else:
            raise ValueError("Failed to load Alpaca account as TradeAccount")

    def execute_trades(
        self, actions: pd.DataFrame, positions: dict[str, deque[Position]]
    ) -> tuple[pd.DataFrame, float]:
        logging.info("Executing trades via Alpaca client: %s", actions)
        for sym, row in actions.iterrows():
            logging.info(
                "sym: %s, size: %s, at price: %s, with signal: %s",
                sym,
                row.get("size"),
                row.get("price"),
                row.get("action"),
            )
            # ! for now we are placing market orders only, limit orders should be used when running live
            # ! and stop loss can easily be added as well
            order = MarketOrderRequest(
                symbol=sym,
                qty=abs(row.get("size", 0)),
                side="buy" if row.get("size", 0) > 0 else "sell",
                type="market",
                time_in_force="day",
            )
            self.client.submit_order(order)

        self._positions = positions

        return actions, actions.get("profit", 0.0).sum()
