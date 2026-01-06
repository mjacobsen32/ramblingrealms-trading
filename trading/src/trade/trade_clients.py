import io
import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import Base

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from alpaca.trading.enums import AccountStatus
from alpaca.trading.models import Position as AlpacaPosition
from alpaca.trading.models import TradeAccount as AlpacaTradeAccount
from alpaca.trading.requests import MarketOrderRequest
from pydantic import BaseModel
from tests.test_records import orders

from trading.cli.trading.trade_config import BrokerType, RRTradeConfig
from trading.src.portfolio.position import (
    PortfolioStats,
    Position,
    PositionDecoder,
    PositionEncoder,
    portfolio_schema,
    positions_schema,
)
from trading.src.user_cache.user_cache import UserCache


class TradeAccount(AlpacaTradeAccount):
    initial_cash: float | None = None


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
        self.config: RRTradeConfig = config
        self.alpaca_account_client = alpaca_account_client
        self.defer_trade_execution = config.defer_trade_execution
        self._account = self.account

    def close(
        self,
        closed_positions: list[Position],
        open_positions: dict[str, deque[Position]],
        pf_history: list[PortfolioStats],
        cash: float,
    ) -> None:
        self._write_open_positions(open_positions=open_positions)
        self._write_closed_positions(closed_positions=closed_positions)
        self._write_pf_stats(stats=pf_history)
        self._write_account(cash)

    def get_clock(self):
        return self.alpaca_account_client.get_clock()

    def get_calendar(self):
        return self.alpaca_account_client.get_calendar()

    @property
    def account(self) -> TradeAccount:
        if hasattr(self, "_account"):
            return self._account
        else:
            self._account = self._load_account()
        return self._account

    @property
    def positions(self) -> dict[str, deque[Position]]:
        if hasattr(self, "_positions"):
            return self._positions
        else:
            self._positions: dict[str, deque[Position]] = self._load_positions()
            return self._positions

    @abstractmethod
    def _load_positions(self) -> dict[str, deque[Position]]:
        raise NotImplementedError

    @abstractmethod
    def _load_account(self) -> TradeAccount:
        raise NotImplementedError

    def _load_pf_stats(self) -> list[PortfolioStats]:
        """
        This is not implemented for Alpaca cause the api manages these and it's just for historical tracking

        :param self: Description
        :return: Description
        :rtype: list[Position]
        """
        return []

    def _load_closed_positions(self) -> list[Position]:
        """
        This is not implemented for Alpaca cause the api manages these and it's just for historical tracking

        :param self: Description
        :return: Description
        :rtype: list[Position]
        """
        return []

    def _write_open_positions(self, open_positions: dict[str, deque[Position]]) -> None:
        pass

    def _write_account(self, cash: float) -> None:
        pass

    def _write_pf_stats(self, stats: list[PortfolioStats]) -> None:
        pass

    def _write_closed_positions(self, closed_positions: list[Position]) -> None:
        pass

    def execute_trades(
        self, actions: pd.DataFrame
    ) -> tuple[pd.DataFrame, float, list[MarketOrderRequest]]:
        orders: list[MarketOrderRequest] = []
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
            orders.append(order)

        return actions, actions["profit"].sum(), orders  # Optional to implement


class LocalTradingClient(TradingClient):
    def __init__(self, config: RRTradeConfig, alpaca_account_client: Any, live: bool):
        self.positions_path = self._resolve_path(config.positions_path, "positions")
        self.account_path = self._resolve_path(config.account_path, "account")
        self.closed_positions_path = self._resolve_path(
            config.closed_positions_path, "closed_positions"
        )
        self.account_value_series_path = self._resolve_path(
            config.account_value_series_path, "account_value_series"
        )
        super().__init__(
            live=live, config=config, alpaca_account_client=alpaca_account_client
        )
        logging.info("Initialized Local Trading Client")

    @staticmethod
    def _resolve_path(path_field, default_name: str) -> Path:
        if path_field is not None:
            return path_field.as_path()
        raise ValueError(f"No path configured for local trading {default_name} data")

    def _write_open_positions(self, open_positions: dict[str, deque[Position]]) -> None:
        self.positions_path.parent.mkdir(parents=True, exist_ok=True)
        self.positions_path.write_text(
            json.dumps(open_positions, default=_json_default, indent=2)
        )
        logging.info("Wrote local positions to %s", self.positions_path)

    def _write_account(self, cash: float) -> None:
        self._account.cash = str(cash)
        self.account_path.parent.mkdir(parents=True, exist_ok=True)
        self.account_path.write_text(self._account.model_dump_json(indent=2))
        logging.info("Wrote local account to %s", self.account_path)

    def _write_pf_stats(self, stats: list[PortfolioStats]) -> None:
        self.account_value_series_path.parent.mkdir(parents=True, exist_ok=True)
        if self.account_value_series_path.suffix == ".parquet":
            df = pd.DataFrame(stats, columns=PortfolioStats.COLS)
            table = pa.Table.from_pandas(df, schema=portfolio_schema)
            pq.write_table(
                table=table,
                where=self.account_value_series_path,
                compression="snappy",
            )
        elif self.account_value_series_path.suffix == ".csv":
            df = pd.DataFrame(stats, columns=PortfolioStats.COLS)
            df.to_csv(self.account_value_series_path, index=False)
        else:
            raise ValueError(
                f"Unsupported file format for portfolio stats: {self.account_value_series_path.suffix}"
            )
        logging.info(
            "Wrote local portfolio stats to %s", self.account_value_series_path
        )

    def _write_closed_positions(self, closed_positions: list[Position]) -> None:
        self.closed_positions_path.parent.mkdir(parents=True, exist_ok=True)
        if self.closed_positions_path.suffix == ".parquet":
            df = pd.DataFrame(closed_positions, columns=Position.COLS)
            table = pa.Table.from_pandas(
                df, schema=positions_schema, preserve_index=False
            )
            pq.write_table(
                table=table,
                where=self.closed_positions_path,
                compression="snappy",
            )
        elif self.closed_positions_path.suffix == ".csv":
            df = pd.DataFrame(closed_positions, columns=Position.COLS)
            df.to_csv(self.closed_positions_path, index=False)
        else:
            raise ValueError(
                f"Unsupported file format for closed positions: {self.closed_positions_path.suffix}"
            )
        logging.info("Wrote local closed positions to %s", self.closed_positions_path)

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
                initial_cash=init_cash,
            )
        else:
            account = TradeAccount.model_validate_json(self.account_path.read_text())

        logging.info("Loaded account id: %s; cash: %s", account.id, account.cash)
        return account

    def _load_pf_stats(self) -> list[PortfolioStats]:
        if not self.account_value_series_path.exists():
            logging.info(
                "No portfolio stats file found at %s", self.account_value_series_path
            )
            return []
        if self.account_value_series_path.suffix == ".parquet":
            table = pq.read_table(
                self.account_value_series_path, schema=portfolio_schema
            )
            stats = [PortfolioStats(**row) for row in table.to_pylist()]
        elif self.account_value_series_path.suffix == ".csv":
            df = pd.read_csv(self.account_value_series_path)
            stats = [PortfolioStats(**row) for _, row in df.iterrows()]
        logging.info(
            "Loaded %d portfolio stats from %s",
            len(stats),
            self.account_value_series_path,
        )
        return stats

    def _load_closed_positions(self) -> list[Position]:
        if not self.closed_positions_path.exists():
            logging.info(
                "No closed positions file found at %s", self.closed_positions_path
            )
            return []
        if self.closed_positions_path.suffix == ".parquet":
            table = pq.read_table(source=self.closed_positions_path)
            df = table.to_pandas()
            positions = [Position(**row) for _, row in df.iterrows()]
        elif self.closed_positions_path.suffix == ".csv":
            df = pd.read_csv(filepath_or_buffer=self.closed_positions_path)
            for col in ["enter_date", "exit_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            positions = [Position(**row) for _, row in df.iterrows()]
        logging.info(
            "Loaded %d closed positions from %s",
            len(positions),
            self.closed_positions_path,
        )
        return positions


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
        self.open_positions_key = str(config.positions_path)
        self.account_key = str(config.account_path)
        self.closed_positions_key = str(config.closed_positions_path)
        self.account_stats_key = str(config.account_value_series_path)

        super().__init__(
            live=True, config=config, alpaca_account_client=alpaca_account_client
        )

        logging.info("RemoteTradingClient Initialized with S3 client")

    def _load_positions(self) -> dict[str, deque[Position]]:
        ret = dict[str, deque[Position]]()
        try:
            positions_response = self._client.get_object(
                Bucket=self.config.bucket_name,
                Key=self.open_positions_key,
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
                    initial_cash=init_cash,
                )
        logging.info("Loaded account id: %s; cash: %s", account.id, account.cash)
        return account

    def _load_pf_stats(self) -> list[PortfolioStats]:
        try:
            stats_response = self._client.get_object(
                Bucket=self.config.bucket_name,
                Key=self.account_stats_key,
            )
            body = stats_response["Body"].read()
            table = pq.read_table(source=io.BytesIO(body), schema=portfolio_schema)
            stats = [PortfolioStats(**row) for row in table.to_pylist()]
            logging.info(
                "Loaded %d remote portfolio stats from bucket %s",
                len(stats),
                self.config.bucket_name,
            )
            return stats
        except Exception as e:
            logging.warning(
                "Failed to load remote portfolio stats from bucket %s: %s",
                self.config.bucket_name,
                str(e),
            )
            logging.info("Initializing new remote portfolio stats")
            return []

    def _load_closed_positions(self) -> list[Position]:
        try:
            closed_positions_response = self._client.get_object(
                Bucket=self.config.bucket_name,
                Key=self.closed_positions_key,
            )
            body = closed_positions_response["Body"].read()
            table = pq.read_table(source=io.BytesIO(body))
            df = table.to_pandas()
            positions: list[Position] = [Position(**row) for _, row in df.iterrows()]
            logging.info(
                "Loaded %d remote closed positions from bucket %s",
                len(positions),
                self.config.bucket_name,
            )
            return positions
        except Exception as e:
            logging.warning(
                "Failed to load remote closed positions from bucket %s: %s",
                self.config.bucket_name,
                str(e),
            )
            logging.info("Initializing new remote closed positions")
            return []

    def _write_closed_positions(self, closed_positions: list[Position]) -> None:
        sink = io.BytesIO()
        df = pd.DataFrame(closed_positions, columns=Position.COLS)
        table = pa.Table.from_pandas(df, schema=positions_schema, preserve_index=False)
        pq.write_table(
            table=table,
            where=sink,
            compression="snappy",
        )
        self._client.put_object(
            Bucket=self.config.bucket_name,
            Key=self.closed_positions_key,
            Body=sink.getvalue(),
            ContentType="application/parquet",
        )
        logging.info(
            "Wrote remote closed positions to bucket %s", self.config.bucket_name
        )

    def _write_account(self, cash: float) -> None:
        self._account.cash = str(cash)
        self._client.put_object(
            Bucket=self.config.bucket_name,
            Key=self.account_key,
            Body=self._account.model_dump_json(indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logging.info("Wrote remote account to bucket %s", self.config.bucket_name)

    def _write_open_positions(self, open_positions: dict[str, deque[Position]]) -> None:
        self._client.put_object(
            Bucket=self.config.bucket_name,
            Key=self.open_positions_key,
            Body=json.dumps(open_positions, default=_json_default, indent=2).encode(
                "utf-8"
            ),
            ContentType="application/json",
        )
        logging.info("Wrote remote positions to bucket %s", self.config.bucket_name)

    def _write_pf_stats(self, stats: list[PortfolioStats]) -> None:
        sink = io.BytesIO()
        stats_dicts = [stat.model_dump() for stat in stats]
        table = pa.Table.from_pylist(stats_dicts, schema=portfolio_schema)
        pq.write_table(table, sink)
        self._client.put_object(
            Bucket=self.config.bucket_name,
            Key=self.account_stats_key,
            Body=sink.getvalue(),
            ContentType="application/parquet",
        )
        logging.info(
            "Wrote remote portfolio stats to bucket %s", self.config.bucket_name
        )


# TODO implement batched orders
class AlpacaClient(TradingClient):
    def __init__(
        self,
        config: RRTradeConfig,
        alpaca_account_client: Any = None,
        live: bool = False,
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
        if hasattr(self, "_account"):
            return self._account
        self._account: TradeAccount = self.alpaca_account_client.get_account()
        return self._account

    def execute_trades(
        self,
        actions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, float, list[MarketOrderRequest]]:
        logging.info(
            "Executing %d trades via Alpaca client", len(actions[actions["size"] != 0])
        )
        order_responses: list[MarketOrderRequest] = []
        actions, total_profit, orders = super().execute_trades(actions)
        for order in orders:
            order_response = self.alpaca_account_client.submit_order(order)
            order_responses.append(order_response)
            logging.info("Submitting order: %s", order)

        return actions, total_profit, order_responses
