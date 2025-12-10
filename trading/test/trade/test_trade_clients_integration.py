import os
from collections import deque
from pathlib import Path
from uuid import UUID

import boto3
import pandas as pd
import pytest
from alpaca.trading.enums import AccountStatus
from alpaca.trading.models import TradeAccount
from moto import mock_aws

from trading.src.portfolio.position import Position, PositionManager
from trading.src.trade.trade_clients import (
    AlpacaClient,
    LocalTradingClient,
    RemoteTradingClient,
)
from trading.src.user_cache.user_cache import UserCache
from trading.test.alg.test_fixtures import *
from trading.test.conftest import *

CONFIG_DIR = Path(__file__).parent.parent / "configs"


class FakeSecret:
    def __init__(self, value: str):
        self.value = value

    def get_secret_value(self) -> str:
        return self.value


def test_position_manager_from_client_populates_holdings():
    class FakeClient:
        @property
        def positions(self) -> dict[str, deque[Position]]:
            return {
                "ABC": deque(
                    [
                        Position(
                            symbol="ABC",
                            lot_size=2,
                            enter_price=10,
                            enter_date=pd.Timestamp("2024-01-01"),
                        )
                    ]
                )
            }

        @property
        def account(self) -> TradeAccount:
            return TradeAccount(
                id=UUID("12345678-1234-5678-1234-567812345678"),
                account_number="acct",
                status=AccountStatus.ACTIVE,
                cash="20.0",
            )

    pm = PositionManager.from_client(FakeClient(), symbols=["ABC", "XYZ"])
    assert pm.df.loc["ABC", "holdings"] == 2
    assert pm.df.loc["ABC", "position_counts"] == 1
    assert pm.df.loc["XYZ", "holdings"] == 0
    assert pm.net_value() == pytest.approx(20.0)


def test_local_client_read_write_positions(
    tmp_path, alpaca_trading_client_mock, local_trade_config
):
    local_trade_config.positions_path.path = str(tmp_path / "positions.json")
    local_trade_config.account_path.path = str(tmp_path / "account.json")
    client = LocalTradingClient(
        local_trade_config, alpaca_trading_client_mock, live=False
    )

    positions = client.positions
    assert len(positions) == 0
    assert int(client.account.cash) == 1000000

    positions = {"AAPL": deque()}
    positions["AAPL"].append(
        Position(
            symbol="AAPL",
            lot_size=5,
            enter_price=100,
            enter_date=pd.Timestamp("2024-01-01"),
        )
    )
    actions = pd.DataFrame(data={"profit": [0]}, index=["AAPL"])
    client.execute_trades(actions=actions, positions=positions)

    assert len(client.positions) == 1
    assert client.positions["AAPL"][0].lot_size == 5

    assert os.path.exists(str(local_trade_config.account_path))
    assert os.path.exists(str(local_trade_config.positions_path))

    assert len(client._load_positions()) == 1


os.environ["MOTO_S3_CUSTOM_ENDPOINTS"] = (
    "https://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.r2.cloudflarestorage.com"
)


@mock_aws
def test_remote_client_uses_s3_store(
    remote_trade_config, alpaca_trading_client_mock, rr_trading_user_cache_path
) -> None:
    cache = UserCache().load()
    conn = boto3.resource(
        "s3",
        endpoint_url=cache.r2_endpoint_url,
    )
    conn.create_bucket(Bucket="rr-storage")

    client = RemoteTradingClient(
        remote_trade_config, alpaca_trading_client_mock, live=False
    )

    positions = client.positions
    assert len(positions) == 0
    assert int(client.account.cash) == 1000000

    positions = {"AAPL": deque()}
    positions["AAPL"].append(
        Position(
            symbol="AAPL",
            lot_size=5,
            enter_price=100,
            enter_date=pd.Timestamp("2024-01-01"),
        )
    )
    actions = pd.DataFrame(data={"profit": [0]}, index=["AAPL"])
    client.execute_trades(actions=actions, positions=positions)

    assert len(client.positions) == 1
    assert client.positions["AAPL"][0].lot_size == 5

    assert (
        conn.Object(bucket_name="rr-storage", key=remote_trade_config.account_path)
        is not None
    )
    assert (
        conn.Object(bucket_name="rr-storage", key=remote_trade_config.positions_path)
        is not None
    )


def test_alpaca_client_positions_and_orders(
    alpaca_trading_client_mock, alpaca_trade_config
):
    client = AlpacaClient(alpaca_trade_config, alpaca_trading_client_mock, live=False)

    positions = client.positions
    assert len(positions) == 2
    assert float(client.account.cash) == 10000.0

    actions, profit = client.execute_trades(
        actions=pd.DataFrame(data={"profit": [0], "size": [100]}, index=["GOOGL"]),
        positions=positions,
    )
