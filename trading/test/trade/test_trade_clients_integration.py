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

from trading.src.portfolio.position import PortfolioStats, Position, PositionManager
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


@pytest.fixture
def open_positions() -> dict[str, deque[Position]]:
    return {
        "AAPL": deque(
            [
                Position(
                    symbol="AAPL",
                    lot_size=5,
                    enter_price=150,
                    enter_date=pd.Timestamp("2024-01-01"),
                )
            ]
        ),
        "GOOGL": deque(
            [
                Position(
                    symbol="GOOGL",
                    lot_size=5,
                    enter_price=2500,
                    enter_date=pd.Timestamp("2024-02-01"),
                )
            ]
        ),
    }


@pytest.fixture
def closed_positions() -> list[Position]:
    return [
        Position(
            symbol="MSFT",
            lot_size=0,
            enter_price=200,
            enter_date=pd.Timestamp("2023-12-01"),
            exit_price=220,
            exit_date=pd.Timestamp("2024-03-01"),
        )
    ]


@pytest.fixture
def pf_history() -> list[PortfolioStats]:
    return [
        PortfolioStats(
            date=str(pd.Timestamp(ts_input="2024-01-01")),
            net_value=100000.0,
            cash=90000.0,
            pnl_pct=1.1,
            pnl=1000.0,
            rolling_pnl=1.1,
            rolling_pnl_pct=1.1,
        )
    ]


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
    tmp_path,
    alpaca_trading_client_mock,
    local_trade_config,
    open_positions,
    closed_positions,
    pf_history,
):
    local_trade_config.positions_path.path = str(tmp_path / "positions.json")
    local_trade_config.account_path.path = str(tmp_path / "account.json")
    local_trade_config.closed_positions_path.path = str(
        tmp_path / "closed_positions.parquet"
    )
    local_trade_config.account_value_series_path.path = str(
        tmp_path / "account_value_series.parquet"
    )
    client = LocalTradingClient(
        local_trade_config, alpaca_trading_client_mock, live=False
    )

    positions = client.positions
    assert len(positions) == 0
    assert int(client.account.cash) == 1000000

    actions = pd.DataFrame(data={"profit": [0]}, index=["AAPL"])
    client.execute_trades(actions=actions)

    client.close(
        closed_positions=closed_positions,
        open_positions=open_positions,
        pf_history=pf_history,
        cash=999500,
    )

    assert os.path.exists(str(local_trade_config.account_path))
    assert os.path.exists(str(local_trade_config.positions_path))
    assert os.path.exists(str(local_trade_config.closed_positions_path))
    assert os.path.exists(str(local_trade_config.account_value_series_path))

    account = client._load_account()
    assert float(account.cash) == 999500.0
    assert float(account.initial_cash) == 1000000.0

    positions = client._load_positions()
    assert len(positions) == 2
    assert positions["AAPL"][0].lot_size == 5

    closed_positions_loaded = client._load_closed_positions()
    assert len(closed_positions_loaded) == 1
    assert closed_positions_loaded[0].symbol == "MSFT"

    pf_history_loaded = client._load_pf_stats()
    assert len(pf_history_loaded) == 1
    assert pf_history_loaded[0].net_value == 100000.0


os.environ["MOTO_S3_CUSTOM_ENDPOINTS"] = (
    "https://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.r2.cloudflarestorage.com"
)


@mock_aws
def test_remote_client_uses_s3_store(
    remote_trade_config,
    alpaca_trading_client_mock,
    rr_trading_user_cache_path,
    open_positions,
    closed_positions,
    pf_history,
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

    actions = pd.DataFrame(data={"profit": [0]}, index=["AAPL"])
    client.execute_trades(actions=actions)

    client.close(
        closed_positions=closed_positions,
        open_positions=open_positions,
        pf_history=pf_history,
        cash=999500,
    )

    assert (
        conn.Object(bucket_name="rr-storage", key=remote_trade_config.account_path)
        is not None
    )
    assert (
        conn.Object(
            bucket_name="rr-storage", key=remote_trade_config.closed_positions_path
        )
        is not None
    )
    assert (
        conn.Object(
            bucket_name="rr-storage", key=remote_trade_config.account_value_series_path
        )
        is not None
    )
    assert (
        conn.Object(bucket_name="rr-storage", key=remote_trade_config.positions_path)
        is not None
    )

    account = client._load_account()
    assert float(account.cash) == 999500.0
    assert account.initial_cash == 1000000.0

    positions = client._load_positions()
    assert len(positions) == 2
    assert positions["AAPL"][0].lot_size == 5

    closed_positions_loaded = client._load_closed_positions()
    assert len(closed_positions_loaded) == 1
    assert closed_positions_loaded[0].symbol == "MSFT"

    pf_history_loaded = client._load_pf_stats()
    assert len(pf_history_loaded) == 1
    assert pf_history_loaded[0].net_value == 100000.0


def test_alpaca_client_positions_and_orders(
    alpaca_trading_client_mock, alpaca_trade_config
):
    client = AlpacaClient(alpaca_trade_config, alpaca_trading_client_mock, live=False)

    positions = client.positions
    assert len(positions) == 2
    assert float(client.account.cash) == 10000.0

    actions, profit = client.execute_trades(
        actions=pd.DataFrame(data={"profit": [0], "size": [100]}, index=["GOOGL"])
    )
