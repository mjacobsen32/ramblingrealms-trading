import io
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from trading.cli.alg.config import ProjectPath
from trading.cli.trading.trade_config import BrokerType, RRTradeConfig
from trading.src.portfolio.position import Position, PositionManager
from trading.src.trade import trade_clients as tc
from trading.src.trade.trade_clients import (
    AlpacaClient,
    LocalTradingClient,
    RemoteTradingClient,
)


class FakeSecret:
    def __init__(self, value: str):
        self.value = value

    def get_secret_value(self) -> str:
        return self.value


def test_position_manager_from_client_populates_holdings():
    class FakeClient:
        def get_positions(self):
            return {
                "ABC": [
                    Position(
                        symbol="ABC",
                        lot_size=2,
                        enter_price=10,
                        enter_date=pd.Timestamp("2024-01-01"),
                    )
                ]
            }

    pm = PositionManager.from_client(FakeClient(), symbols=["ABC", "XYZ"])
    assert pm.df.loc["ABC", "holdings"] == 2
    assert pm.df.loc["ABC", "position_counts"] == 1
    assert pm.df.loc["XYZ", "holdings"] == 0
    assert pm.net_value() == pytest.approx(20.0)


def test_local_client_read_write_positions(tmp_path):
    positions_path = tmp_path / "positions.json"
    account_path = tmp_path / "account.json"
    positions_path.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "AAPL",
                        "qty": 3,
                        "avg_entry_price": 5,
                        "enter_date": "2024-01-01",
                    }
                ]
            }
        )
    )
    cfg = RRTradeConfig(
        broker=BrokerType.LOCAL,
        positions_path=ProjectPath(path=str(positions_path)),
        account_path=ProjectPath(path=str(account_path)),
    )
    client = LocalTradingClient(cfg)

    positions = client.get_positions()
    assert positions["AAPL"][0].lot_size == 3

    actions = pd.DataFrame(
        [
            {
                "size": 1,
                "price": 10,
                "timestamp": pd.Timestamp("2024-02-01"),
            }
        ],
        index=["AAPL"],
    )
    client.execute_trades(actions)
    saved = json.loads(positions_path.read_text())
    assert len(saved["positions"]) == 2


def test_remote_client_uses_s3_store(monkeypatch) -> None:
    store: dict[tuple[str, str], bytes] = {}

    class FakeUserCache:
        r2_access_key_id = FakeSecret("id")
        r2_secret_access_key = FakeSecret("secret")
        r2_endpoint_url = "http://example"

        @classmethod
        def load(cls):
            return cls()

    class FakeS3Client:
        def __init__(self, backing):
            self.backing = backing

        def get_object(self, Bucket, Key):
            key = (Bucket, Key)
            if key not in self.backing:
                raise Exception("missing")
            return {"Body": io.BytesIO(self.backing[key])}

        def put_object(self, Bucket, Key, Body):
            self.backing[(Bucket, Key)] = Body
            return {}

    class FakeBoto3:
        def __init__(self, backing):
            self.backing = backing

        def client(self, *_args, **_kwargs):
            return FakeS3Client(self.backing)

    initial_positions = {
        "positions": [
            {
                "symbol": "MSFT",
                "qty": 4,
                "avg_entry_price": 7,
                "enter_date": "2024-01-01",
            }
        ]
    }
    store[("bucket", "positions.json")] = json.dumps(initial_positions).encode()

    monkeypatch.setattr(tc, "UserCache", FakeUserCache)
    monkeypatch.setattr(tc, "boto3", FakeBoto3(store))

    cfg = RRTradeConfig(
        broker=BrokerType.REMOTE,
        remote_bucket="bucket",
        remote_prefix="",
    )
    client = RemoteTradingClient(cfg)
    positions = client.get_positions()
    assert positions["MSFT"][0].lot_size == 4

    actions = pd.DataFrame(
        [
            {
                "size": 2,
                "price": 9,
                "timestamp": pd.Timestamp("2024-02-01"),
            }
        ],
        index=["MSFT"],
    )
    client.execute_trades(actions, positions)
    saved = json.loads(store[("bucket", "positions.json")])
    assert len(saved["positions"]) == 2


def test_alpaca_client_positions_and_orders(monkeypatch):
    orders = []

    class FakeAlpacaClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_account(self):
            return SimpleNamespace(account_number="acct")

        def get_all_positions(self):
            return [
                SimpleNamespace(qty="3", avg_entry_price="10", symbol="XYZ"),
            ]

        def submit_order(self, order):
            orders.append(order)

    class FakeOrderSide:
        BUY = "buy"
        SELL = "sell"

    class FakeOrderType:
        MARKET = "market"

    class FakeTIF:
        GTC = "gtc"

    class FakeOrderRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCache:
        alpaca_api_key = FakeSecret("key")
        alpaca_api_secret = FakeSecret("secret")
        alpaca_api_key_live = FakeSecret("lkey")
        alpaca_api_secret_live = FakeSecret("lsecret")

        @classmethod
        def load(cls):
            return cls()

    monkeypatch.setattr(tc, "AlpacaTradingClient", FakeAlpacaClient)
    monkeypatch.setattr(tc, "OrderSide", FakeOrderSide)
    monkeypatch.setattr(tc, "OrderType", FakeOrderType)
    monkeypatch.setattr(tc, "TimeInForce", FakeTIF)
    monkeypatch.setattr(tc, "OrderRequest", FakeOrderRequest)
    monkeypatch.setattr(tc, "UserCache", FakeCache)

    cfg = RRTradeConfig(broker=BrokerType.ALPACA)
    client = AlpacaClient(cfg, live=False)
    positions = client.get_positions()
    assert positions["XYZ"][0].lot_size == 3

    actions = pd.DataFrame(
        [
            {
                "size": 2,
                "price": 10,
                "timestamp": pd.Timestamp("2024-03-01"),
            }
        ],
        index=["XYZ"],
    )
    client.execute_trades(actions)
    assert len(orders) == 1
    assert orders[0].kwargs["symbol"] == "XYZ"
