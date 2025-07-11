from pathlib import Path

import pytest

from trading.cli.alg.alg import backtest, train
from trading.src.alg.backtest.backtesting import BackTesting

CONFIG_DIR = Path(__file__).parent.parent / "configs"

from unittest.mock import patch


def test_train_backtest():
    train(config=str(CONFIG_DIR / "generic_alg.json"), dry_run=False, no_test=False)
    backtest(config=str(CONFIG_DIR / "generic_alg.json"))
