import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
from alpaca.trading.enums import AccountStatus
from alpaca.trading.models import Position as AlpacaPosition
from alpaca.trading.models import TradeAccount
from alpaca.trading.requests import MarketOrderRequest

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
    def from_config(
        cls, config: RRTradeConfig, alpaca_account_client: Any, live: bool = False
    ) -> "TradingClient":
        if config.broker == BrokerType.ALPACA:
            return AlpacaClient(
                config=config, alpaca_account_client=alpaca_account_client, live=live
            )
        elif config.broker == BrokerType.LOCAL:
            return LocalTradingClient(
                config=config, alpaca_account_client=alpaca_account_client, live=live
            )
        elif config.broker == BrokerType.REMOTE:
            return RemoteTradingClient(
                config=config, alpaca_account_client=alpaca_account_client, live=live
            )
        else:
            raise ValueError(f"Unsupported broker type: {config.broker}")

    def __init__(self, live: bool, config: RRTradeConfig, alpaca_account_client: Any):
        self.live = live
        self.config = config
        self.alpaca_account_client = alpaca_account_client

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
    def __init__(self, config: RRTradeConfig, alpaca_account_client: Any, live: bool):
        self.positions_path = self._resolve_path(config.positions_path, "positions")
        self.account_path = self._resolve_path(config.account_path, "account")
        super().__init__(
            live=live, config=config, alpaca_account_client=alpaca_account_client
        )
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

        return actions, actions["profit"].sum()


class RemoteTradingClient(TradingClient):
    def __init__(self, config: RRTradeConfig, alpaca_account_client: Any, live: bool):
        logging.info("Initialized Remote Trading Client")
        cache = UserCache().load()
        config.broker_kwargs.update(
            {
                "aws_access_key_id": cache.r2_access_key_id.get_secret_value(),
                "aws_secret_access_key": cache.r2_secret_access_key.get_secret_value(),
                "endpoint_url": cache.r2_endpoint_url,
            }
        )

        self._client = boto3.client(**config.broker_kwargs)
        self.positions_key = str(config.positions_path)
        self.account_key = str(config.account_path)

        super().__init__(
            live=True, config=config, alpaca_account_client=alpaca_account_client
        )

        logging.info("RemoteTradingClient Initialized with S3 client")

    def __del__(self) -> None:
        if self.config.defer_trade_execution:
            self._write_account()
            self._write_positions()

    def _write_account(self) -> None:
        self._client.put_object(
            Bucket=self.config.bucket_name,
            Key=self.account_key,
            Body=self.account.model_dump_json(indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logging.info("Wrote remote account to bucket %s", self.config.bucket_name)

    def _write_positions(self) -> None:
        self._client.put_object(
            Bucket=self.config.bucket_name,
            Key=self.positions_key,
            Body=json.dumps(self.positions, default=_json_default, indent=2).encode(
                "utf-8"
            ),
            ContentType="application/json",
        )
        logging.info("Wrote remote positions to bucket %s", self.config.bucket_name)

    def _load_positions(self) -> dict[str, deque[Position]]:
        ret = dict[str, deque[Position]]()
        try:
            positions_response = self._client.get_object(
                Bucket=self.config.bucket_name,
                Key=self.positions_key,
            )
            data = json.loads(positions_response["Body"].read().decode("utf-8"))
            count = 0
            for key, positions in data.items():
                for pos in positions:
                    count += 1
                    ret.setdefault(key, deque()).append(
                        json.loads(json.dumps(pos), object_hook=PositionDecoder)
                    )
            logging.info(
                "Loaded %d remote positions from bucket %s",
                count,
                self.config.bucket_name,
            )
            return ret
        except Exception as e:
            logging.warning(
                "Failed to load remote positions from bucket %s: %s",
                self.config.bucket_name,
                str(e),
            )
            logging.info("Initializing new remote positions")
            return ret

    def _load_account(self) -> TradeAccount:
        try:
            account = TradeAccount.model_validate_json(
                self._client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=self.account_key,
                )["Body"]
                .read()
                .decode("utf-8")
            )
            logging.info(
                "Loaded remote account from bucket %s", self.config.bucket_name
            )
        except Exception as e:
            logging.warning(
                "Failed to load remote account from bucket %s: %s",
                self.config.bucket_name,
                str(e),
            )
            logging.info("Initializing new remote account")
            if self.config.portfolio_config is None:
                raise ValueError(
                    "Portfolio config is required for local account initialization"
                )
            else:
                init_cash = self.config.portfolio_config.initial_cash
                account = TradeAccount(
                    id=self.config.id,
                    account_number=self.config.account_number,
                    status=AccountStatus.ACTIVE,
                    cash=str(init_cash),
                )
        logging.info("Loaded account id: %s; cash: %s", account.id, account.cash)
        return account

    def execute_trades(
        self, actions: pd.DataFrame, positions: dict[str, deque[Position]]
    ) -> tuple[pd.DataFrame, float]:
        self._positions = positions
        if self.config.defer_trade_execution:
            logging.info("Deferring trade execution; not uploading trades")
        else:
            logging.info("Uploading trades to remote storage")
            self._write_account()
            self._write_positions()
        return actions, actions["profit"].sum()


# TODO implement batched orders
class AlpacaClient(TradingClient):
    def __init__(
        self,
        config: RRTradeConfig,
        live: bool = False,
        alpaca_account_client: Any = None,
    ):
        super().__init__(
            live=live, config=config, alpaca_account_client=alpaca_account_client
        )

    def _load_positions(self) -> dict[str, deque[Position]]:
        ret = dict[str, deque[Position]]()
        alpaca_positions = self.alpaca_account_client.get_all_positions()
        for alpaca_pos in alpaca_positions:
            if isinstance(alpaca_pos, AlpacaPosition):
                pos = Position.from_alpaca_position(alpaca_pos)
            else:
                raise ValueError("Failed to load Alpaca position as Position")
            ret.setdefault(pos.symbol, deque()).append(pos)
        logging.info("Loaded %d positions from Alpaca", len(alpaca_positions))
        return ret

    def _load_account(self) -> TradeAccount:
        account = self.alpaca_account_client.get_account()
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
        logging.info(
            "Executing %d trades via Alpaca client", len(actions[actions["size"] != 0])
        )
        for sym, row in actions.iterrows():
            logging.info(
                "sym: %s, size: %s, at price: %s, with signal: %s",
                sym,
                row.get("size"),
                row.get("price"),
                row.get("action"),
            )
            if row.get("size", 0) == 0:
                continue
            # ! for now we are placing market orders only, limit orders should be used when running live
            # ! and stop loss can easily be added as well
            order = MarketOrderRequest(
                symbol=sym,
                qty=abs(row.get("size", 0)),
                side="buy" if row.get("size", 0) > 0 else "sell",
                type="market",
                time_in_force="day",
            )
            self.alpaca_account_client.submit_order(order)
            logging.info("Submitting order: %s", order)

        self._positions = positions

        return actions, actions["profit"].sum()
