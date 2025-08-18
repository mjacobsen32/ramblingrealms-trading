from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from trading.cli.alg.alg import backtest, train
from trading.cli.alg.config import ProjectPath
from trading.cli.main import app
from trading.src.alg.backtest.backtesting import BackTesting

CONFIG_DIR = Path(__file__).parent.parent / "configs"

import os
import re
import shutil
import tempfile
from unittest.mock import patch

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


def test_train_backtest_analysis(temp_dirs):
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
