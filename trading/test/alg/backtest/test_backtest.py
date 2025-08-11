import pandas as pd
import pytest

from trading.test.alg.test_fixtures import *


def test_backtest_run(backtest):
    pf = backtest.run()
    stats = pf.stats()

    assert str(stats["Start"]) == "2023-10-19 04:00:00+00:00"
    assert str(stats["End"]) == "2023-12-27 05:00:00+00:00"
    assert stats["Period"] == pd.Timedelta(days=48)
    assert stats["Total Trades"] > 0

    assert len(pf.orders()) > 0
    assert len(pf.trades()) > 0
