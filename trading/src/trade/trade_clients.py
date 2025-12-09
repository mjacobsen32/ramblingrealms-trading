import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient as AlpacaTradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import OrderRequest

try:
    import boto3
except Exception:  # pragma: no cover - optional dependency for remote client
    boto3 = None

from trading.cli.trading.trade_config import BrokerType, RRTradeConfig
from trading.src.portfolio.position import Position
from trading.src.user_cache.user_cache import UserCache


def _json_default(obj: object):  # -> str | Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, deque):
        return list(obj)
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

    def __init__(self, live: bool = False, config: RRTradeConfig | None = None):
        self.live = live
        self.config = config
        user_cache = UserCache().load()
        self.alpaca_api_key = user_cache.alpaca_api_key
        self.alpaca_api_secret = user_cache.alpaca_api_secret

        self.alpaca_account_client: AlpacaTradingClient = AlpacaTradingClient(
            self.alpaca_api_key.get_secret_value(),
            self.alpaca_api_secret.get_secret_value(),
            paper=not live,
        )

    def get_prices(self, symbols: list[str]) -> pd.Series:
        prices = {}
        for sym in symbols:
            try:
                barset = self.alpaca_account_client.get_bars(
                    symbol=sym,
                    timeframe="1Day",
                    limit=1,
                    adjustment="raw",
                )
                if barset and len(barset) > 0:
                    bars = barset[sym]
                    if bars and len(bars) > 0:
                        prices[sym] = bars[-1].c  # closing price of the latest bar
            except Exception as exc:  # pragma: no cover - network
                logging.error("Failed to fetch price for %s: %s", sym, exc)
        return pd.Series(prices)

    @abstractmethod
    def get_account(self):
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> dict[str, deque[Position]]:
        raise NotImplementedError

    @abstractmethod
    def execute_trades(
        self, actions: pd.DataFrame, positions: dict[str, deque[Position]]
    ) -> tuple[pd.DataFrame, float]:
        raise NotImplementedError

    def state(self, symbols: list[str]) -> np.ndarray:
        raise NotImplementedError("This method should be implemented in a subclass.")

    def get_clock(self):
        return self.alpaca_account_client.get_clock()

    def get_calendar(self):
        return self.alpaca_account_client.get_calendar()


class LocalTradingClient(TradingClient):
    def __init__(self, config: RRTradeConfig):
        super().__init__(live=False, config=config)
        logging.info("Initialized Local Trading Client")
        self.positions_path = self._resolve_path(config.positions_path, "positions")
        self.account_path = self._resolve_path(config.account_path, "account")

    @staticmethod
    def _resolve_path(path_field, default_name: str) -> Path:
        if path_field is not None:
            return path_field.as_path()
        raise ValueError(f"No path configured for local trading {default_name} data")

    def _load_positions(self) -> dict:
        if not self.positions_path.exists():
            return {"positions": []}
        return json.loads(self.positions_path.read_text())

    def _save_positions(self, payload: dict) -> None:
        self.positions_path.parent.mkdir(parents=True, exist_ok=True)
        self.positions_path.write_text(
            json.dumps(payload, indent=2, default=_json_default)
        )

    def _load_account(self) -> dict:
        if not self.account_path.exists():
            return {}
        return json.loads(self.account_path.read_text())

    def _save_account(self, payload: dict) -> None:
        self.account_path.parent.mkdir(parents=True, exist_ok=True)
        self.account_path.write_text(
            json.dumps(payload, indent=2, default=_json_default)
        )

    def get_account(self):
        return self._load_account()

    def get_positions(self) -> dict[str, deque[Position]]:
        data = self._load_positions()
        positions: dict[str, deque[Position]] = {}
        for raw in data.get("positions", []):
            sym = raw.get("symbol")
            qty = float(raw.get("qty", 0))
            entry = float(raw.get("avg_entry_price", 0))
            enter_date = pd.to_datetime(raw.get("enter_date", pd.Timestamp.utcnow()))
            exit_date = pd.to_datetime(raw.get("exit_date", None))
            exit_size = float(raw.get("exit_size", 0))
            position_type = raw.get("position_type", "long")
            pos = Position(
                symbol=sym,
                lot_size=qty,
                enter_price=entry,
                enter_date=enter_date,
                exit_date=exit_date,
                exit_size=exit_size,
                position_type=position_type,
            )
            positions.setdefault(sym, deque()).append(pos)
        logging.info("Loaded positions: %s", positions)
        return positions

    def execute_trades(
        self, actions: pd.DataFrame, positions: dict[str, deque[Position]]
    ) -> tuple[pd.DataFrame, float]:
        logging.info(
            "size: %s, at price: %s, with signal: %s",
            actions["size"],
            actions["price"],
            actions["action"],
        )
        self._save_positions(positions)

        return actions, actions.get("profit", 0.0).sum()

    def state(self, symbols: list[str]) -> np.ndarray:
        return np.zeros(len(symbols))


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

    def _fetch_remote_json(self, key: str) -> dict:
        obj = self._s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}{key}")
        return json.loads(obj["Body"].read().decode())

    def _put_remote_json(self, key: str, data: dict) -> None:
        self._s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}{key}",
            Body=json.dumps(data, indent=2, default=_json_default).encode(),
        )

    def get_account(self):
        try:
            data = self._fetch_remote_json(self.account_key)
        except Exception:
            data = {}
        return data

    def get_positions(self) -> dict[str, deque[Position]]:
        try:
            data = self._fetch_remote_json(self.positions_key)
        except Exception:
            data = {"positions": []}
        positions: dict[str, deque[Position]] = {}
        for raw in data.get("positions", []):
            sym = raw.get("symbol")
            qty = float(raw.get("qty", 0))
            entry = float(raw.get("avg_entry_price", 0))
            enter_date = pd.to_datetime(raw.get("enter_date", pd.Timestamp.utcnow()))
            pos = Position(
                symbol=sym, lot_size=qty, enter_price=entry, enter_date=enter_date
            )
            positions.setdefault(sym, deque()).append(pos)
        return positions

    def execute_trades(
        self, actions: pd.DataFrame, positions: dict[str, deque[Position]]
    ) -> tuple[pd.DataFrame, float]:
        try:
            data = self._fetch_remote_json(self.positions_key)
        except Exception:
            data = {"positions": []}
        positions_list = data.setdefault("positions", [])
        for sym, row in actions.iterrows():
            positions_list.append(
                {
                    "symbol": sym,
                    "qty": row.get("size", 0),
                    "avg_entry_price": row.get("price", 0),
                    "enter_date": str(row.get("timestamp", pd.Timestamp.utcnow())),
                }
            )
        self._put_remote_json(self.positions_key, data)
        return actions, 0.0

    def state(self, symbols: list[str]) -> np.ndarray:
        return np.zeros(len(symbols))


class AlpacaClient(TradingClient):
    account_details: Any

    def __init__(self, config: RRTradeConfig, live: bool = False):
        super().__init__(live=live, config=config)
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

        self.account_details: Any = self.client.get_account()
        logging.info(
            "Initialized Alpaca Client: %s", self.account_details.account_number
        )

    def get_account(self):
        return self.account_details

    def get_positions(self) -> dict[str, deque[Position]]:
        alpaca_positions = self.client.get_all_positions()
        positions: dict[str, deque[Position]] = {}
        for p in alpaca_positions:
            qty = float(getattr(p, "qty", 0))
            entry = float(getattr(p, "avg_entry_price", 0))
            sym = getattr(p, "symbol", "")
            enter_date = pd.Timestamp.utcnow()
            pos = Position(
                symbol=sym, lot_size=qty, enter_price=entry, enter_date=enter_date
            )
            positions.setdefault(sym, deque()).append(pos)
        return positions

    def execute_trades(
        self, actions: pd.DataFrame, positions: dict[str, deque[Position]]
    ) -> tuple[pd.DataFrame, float]:
        logging.info("Executing trades via Alpaca")
        for sym, row in actions.iterrows():
            size = row.get("size", 0)
            if size == 0:
                continue
            side = OrderSide.BUY if size > 0 else OrderSide.SELL
            qty = abs(float(size))
            order = OrderRequest(
                symbol=sym,
                qty=qty,
                side=side,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
            )
            try:
                self.client.submit_order(order)
            except Exception as exc:  # pragma: no cover - network
                logging.error("Failed to submit order for %s: %s", sym, exc)
        return actions, 0.0
