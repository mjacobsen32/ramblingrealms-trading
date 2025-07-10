import pytest

from trading.test.alg.test_fixtures import *


def test_backtest_run(backtest):
    backtest.run()
    stats = backtest.stats()

    assert str(stats["Start"]) == "2023-01-03 05:00:00+00:00"
    assert str(stats["End"]) == "2023-10-18 04:00:00+00:00"
    assert stats["Period"] == 200
    assert stats["Total Trades"] > 0
