import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trading.cli.alg.config import ProjectPath
from trading.cli.rr_trading import app
from trading.cli.trading.trade_config import RRTradeConfig
from trading.src.trade.trade_api import Trade
from trading.test.mocks.alpaca_trading_client_mock import AlpacaTradingClientMock
from trading.test.mocks.stock_historical_data_client_mock import (
    StockHistoricalDataClientMock,
)

CONFIG_DIR = Path(__file__).parent.parent / "configs"

import os
import shutil
import tempfile

runner = CliRunner()


def setup_module(module):
    temp_cache = tempfile.NamedTemporaryFile(delete=True, delete_on_close=True)
    os.environ["RR_TRADING_USER_CACHE_PATH"] = temp_cache.name


def teardown_module(module):
    temp_cache_path = os.environ.get("RR_TRADING_USER_CACHE_PATH")
    if temp_cache_path and os.path.exists(temp_cache_path):
        os.remove(temp_cache_path)


@pytest.fixture(scope="session", autouse=True)
def temp_dirs():
    yield
    out_dir = Path(str(ProjectPath.OUT_DIR))
    if out_dir.exists():
        shutil.rmtree(out_dir)


@pytest.fixture(scope="session")
def fake_keys(temp_dirs):
    key_path = Path(str(ProjectPath.OUT_DIR)) / "test_key.txt"
    secret_path = Path(str(ProjectPath.OUT_DIR)) / "test_secret.txt"

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text("fake_key")
    secret_path.write_text("fake_secret")
    yield

    key_path.unlink()
    secret_path.unlink()


def test_main():
    result = runner.invoke(app, "--help", color=False)
    assert result.exit_code == 0


def test_print_config():
    result = runner.invoke(app, ["print-config"], color=False)
    assert result.exit_code == 0


def test_setup(fake_keys):
    key_path = Path(str(ProjectPath.OUT_DIR)) / "test_key.txt"
    secret_path = Path(str(ProjectPath.OUT_DIR)) / "test_secret.txt"
    result = runner.invoke(
        app,
        ["setup"],
        color=False,
        input=f"y\n{key_path}\n{secret_path}\nn\ny\n{key_path}\n{secret_path}\nn\ny\n{key_path}\ny\n{key_path}\n{secret_path}\ny\n{key_path}\n{secret_path}\n",
    )
    assert result.exit_code == 0


def test_train_backtest_analysis_trade(temp_dirs):
    train_res = runner.invoke(
        app,
        ["alg", "train", "--config", str(CONFIG_DIR / "generic_alg.json")],
        color=False,
    )
    assert train_res.exit_code == 0
    assert "Training completed successfully." in train_res.output

    backtest_res = runner.invoke(
        app,
        ["alg", "backtest", "--config", str(CONFIG_DIR / "generic_alg.json")],
        color=False,
    )
    assert backtest_res.exit_code == 0
    assert "Backtest results will be saved to: " in backtest_res.output
    assert "Backtesting completed successfully." in backtest_res.output

    analysis_res = runner.invoke(
        app,
        [
            "alg",
            "analysis",
            "--config",
            str(CONFIG_DIR / "generic_alg.json"),
            "-o",
            str(ProjectPath.OUT_DIR),
        ],
        color=False,
    )
    assert analysis_res.exit_code == 0
    assert "Analysis completed successfully." in analysis_res.output


def test_trade_execution() -> None:
    market_data_client = StockHistoricalDataClientMock()

    alpaca_account_client = AlpacaTradingClientMock()

    with Path.open(Path(str(CONFIG_DIR / "trade_config.json"))) as f:
        rr_trade_config = RRTradeConfig.model_validate_json(f.read())

    trade_client = Trade(
        config=rr_trade_config,
        market_data_client=market_data_client,
        alpaca_account_client=alpaca_account_client,
        live=False,
    )
    trade_client.run_model(predict_time=(datetime.datetime.fromisoformat("2025-01-01")))
