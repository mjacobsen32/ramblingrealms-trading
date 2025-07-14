import pytest

from trading.test.alg.test_fixtures import *


def test_backtest_run(backtest):
    pf = backtest.run()
    stats = pf.stats()

    assert str(stats["Start"]) == "2023-10-20 04:00:00+00:00"
    assert str(stats["End"]) == "2023-12-28 05:00:00+00:00"
    assert stats["Period"] == 48
    assert stats["Total Trades"] > 0
