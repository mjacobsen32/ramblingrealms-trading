from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from trading.cli.alg.alg import backtest, train
from trading.cli.alg.config import ProjectPath
from trading.cli.main import app
from trading.src.alg.backtest.backtesting import BackTesting

CONFIG_DIR = Path(__file__).parent.parent / "configs"

import re
import shutil
from unittest.mock import patch

runner = CliRunner()


@pytest.fixture(scope="session", autouse=True)
def temp_dirs():
    yield
    out_dir = Path(str(ProjectPath.OUT_DIR))
    if out_dir.exists():
        shutil.rmtree(out_dir)


def test_main():
    result = runner.invoke(app, "--help")
    assert result.exit_code == 0
    assert "Usage: rr_trading [OPTIONS] COMMAND [ARGS]..." in result.output


def test_train_backtest_analysis(temp_dirs):
    train_res = runner.invoke(
        app, ["alg", "train", "--config", str(CONFIG_DIR / "generic_alg.json")]
    )
    assert train_res.exit_code == 0
    assert "Training completed successfully." in train_res.output

    backtest_res = runner.invoke(
        app, ["alg", "backtest", "--config", str(CONFIG_DIR / "generic_alg.json")]
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
    )
    assert analysis_res.exit_code == 0
    assert "Analysis completed successfully." in analysis_res.output
