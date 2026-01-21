import datetime
from pathlib import Path

import pandas as pd
import pytest
from zoneinfo import ZoneInfo
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


def test_loggers():
    result = runner.invoke(
        app, "--log-level-file INFO --log-level-console INFO --help", color=False
    )
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
        predict_time=datetime.datetime.fromisoformat("2024-12-31"),
        end_predict_time=datetime.datetime.fromisoformat("2025-02-04 05:00:00+00:00"),
        fetch_data=False,
    )

    # TEST_SET starts on 2024-06-05 and ends on 2025-06-04

    # First call should succeed without throwing
    time_series = trade_client.run_model(
        predict_time=datetime.datetime(
            year=2024, month=12, day=31, hour=5, tzinfo=datetime.timezone.utc
        ),
        end_predict_time=datetime.datetime(
            year=2025, month=2, day=4, hour=5, tzinfo=datetime.timezone.utc
        ),
    )
    assert time_series[0]["timestamp"] == pd.Timestamp(
        "2024-12-31 05:00:00+0000", tz="UTC"
    )

    assert time_series[-1]["timestamp"] == pd.Timestamp(
        "2025-02-04 05:00:00+0000", tz="UTC"
    )

    # Weekend request should raise OUT_OF_RANGE error
    with pytest.raises(Trade.LiveTradeError) as exc_info:
        trade_client.run_model(
            predict_time=datetime.datetime(
                year=2025, month=2, day=1, tzinfo=datetime.timezone.utc
            ),
            end_predict_time=datetime.datetime(
                year=2025, month=2, day=1, tzinfo=datetime.timezone.utc
            ),
        )

    assert exc_info.value.error_type == Trade.LiveTradeError.OUT_OF_RANGE

    time_series = trade_client.run_model(
        predict_time=datetime.datetime(
            year=2025, month=2, day=3, hour=5, tzinfo=datetime.timezone.utc
        ),
        end_predict_time=datetime.datetime(
            year=2025, month=2, day=3, hour=5, tzinfo=datetime.timezone.utc
        ),
    )

    assert time_series[0]["timestamp"] == pd.Timestamp(
        "2025-02-03 05:00:00+0000", tz="UTC"
    )

    # 2025 Feb 3rd at 6 AM UTC is midnight EST

    time_series = trade_client.run_model(
        predict_time=datetime.datetime(
            year=2025, month=2, day=3, hour=6, tzinfo=datetime.timezone.utc
        ),
        end_predict_time=datetime.datetime(
            year=2025, month=2, day=3, hour=6, tzinfo=datetime.timezone.utc
        ),
    )

    assert time_series[0]["timestamp"] == pd.Timestamp(
        "2025-02-03 05:00:00+0000", tz="UTC"
    )

    # 2025 Feb 3rd at 4 AM UTC is 11:00 pm EST, so the data returned from alpaca will be from the day prior...

    time_series = trade_client.run_model(
        predict_time=datetime.datetime(
            year=2025, month=2, day=3, hour=4, tzinfo=datetime.timezone.utc
        ),
        end_predict_time=datetime.datetime(
            year=2025, month=2, day=3, hour=4, tzinfo=datetime.timezone.utc
        ),
    )

    assert time_series[0]["timestamp"] == pd.Timestamp(
        "2025-01-31 05:00:00+0000", tz="UTC"
    )

    # 2025 Feb 3rd at 4 AM UTC is 11:00 pm EST, so the data returned from alpaca will be from the day prior...

    five_pm_eastern = pd.Timestamp("2025-02-03 17:00:00-05:00", tz="America/New_York")
    assert five_pm_eastern.tz_convert("UTC").hour == 22

    time_series = trade_client.run_model(
        predict_time=five_pm_eastern.tz_convert("UTC").to_pydatetime(),
        end_predict_time=five_pm_eastern.tz_convert("UTC").to_pydatetime(),
    )

    assert time_series[0]["timestamp"] == pd.Timestamp(
        "2025-02-03 05:00:00+0000", tz="UTC"
    )

    # This is valid date but outside the range of the cache
    with pytest.raises(Trade.LiveTradeError) as exc_info:
        trade_client.run_model(
            predict_time=datetime.datetime(
                year=2025, month=6, day=5, tzinfo=datetime.timezone.utc
            ),
            end_predict_time=datetime.datetime(
                year=2025, month=6, day=5, tzinfo=datetime.timezone.utc
            ),
        )


@pytest.mark.parametrize(
    "predict_time, expected_actual_predict_time",
    [
        (
            datetime.datetime(
                year=2024, month=12, day=31, hour=5, tzinfo=datetime.timezone.utc
            ),
            datetime.datetime(
                year=2024, month=12, day=31, hour=5, tzinfo=datetime.timezone.utc
            ),
        ),
        (
            datetime.datetime(
                year=2024, month=12, day=31, hour=0, tzinfo=ZoneInfo("America/New_York")
            ),
            datetime.datetime(
                year=2024, month=12, day=31, hour=0, tzinfo=ZoneInfo("America/New_York")
            ),
        ),
    ],
)
def test_trade_predict_time(predict_time, expected_actual_predict_time) -> None:
    market_data_client = StockHistoricalDataClientMock()

    alpaca_account_client = AlpacaTradingClientMock()

    with Path.open(Path(str(CONFIG_DIR / "trade_config.json"))) as f:
        rr_trade_config = RRTradeConfig.model_validate_json(f.read())

    for predict_time, expected_actual_predict_time in zip(
        [predict_time], [expected_actual_predict_time]
    ):
        trade_client = Trade(
            config=rr_trade_config,
            market_data_client=market_data_client,
            alpaca_account_client=alpaca_account_client,
            live=False,
            predict_time=predict_time,
            end_predict_time=predict_time,
            fetch_data=False,
        )
        time_series = trade_client.run_model(
            predict_time=predict_time,
            end_predict_time=predict_time,
        )
        assert time_series[0]["timestamp"] == expected_actual_predict_time


@pytest.mark.parametrize(
    "predict_time, end_predict_time, expected_actual_predict_time, length_of_series",
    [
        (
            datetime.datetime(
                year=2024, month=12, day=30, hour=5, tzinfo=datetime.timezone.utc
            ),
            datetime.datetime(
                year=2024, month=12, day=31, hour=5, tzinfo=datetime.timezone.utc
            ),
            datetime.datetime(
                year=2024, month=12, day=31, hour=5, tzinfo=datetime.timezone.utc
            ),
            2,
        ),
        (
            datetime.datetime(
                year=2024, month=12, day=26, hour=0, tzinfo=ZoneInfo("America/New_York")
            ),
            datetime.datetime(
                year=2024, month=12, day=31, hour=0, tzinfo=ZoneInfo("America/New_York")
            ),
            datetime.datetime(
                year=2024, month=12, day=31, hour=0, tzinfo=ZoneInfo("America/New_York")
            ),
            4,
        ),
    ],
)
def test_trade_predict_time_with_endtime(
    predict_time, end_predict_time, expected_actual_predict_time, length_of_series
) -> None:
    market_data_client = StockHistoricalDataClientMock()

    alpaca_account_client = AlpacaTradingClientMock()

    with Path.open(Path(str(CONFIG_DIR / "trade_config.json"))) as f:
        rr_trade_config = RRTradeConfig.model_validate_json(f.read())

    for (
        predict_time,
        end_predict_time,
        expected_actual_predict_time,
        length_of_series,
    ) in zip(
        [predict_time],
        [end_predict_time],
        [expected_actual_predict_time],
        [length_of_series],
    ):
        trade_client = Trade(
            config=rr_trade_config,
            market_data_client=market_data_client,
            alpaca_account_client=alpaca_account_client,
            live=False,
            predict_time=predict_time,
            end_predict_time=end_predict_time,
            fetch_data=False,
        )
        time_series = trade_client.run_model(
            predict_time=predict_time,
            end_predict_time=end_predict_time,
        )
        assert time_series[-1]["timestamp"] == expected_actual_predict_time
        assert len(time_series) == length_of_series
