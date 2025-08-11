import pandas as pd
import pytest

from trading.test.alg.test_fixtures import *


def test_backtest_run(backtest):
    pf = backtest.run()
    pf.as_vbt_pf(df=backtest.env.data)
    stats = pf.stats()

    assert str(stats["Start"]) == "2023-10-19 04:00:00+00:00"
    assert str(stats["End"]) == "2023-12-29 05:00:00+00:00"
    assert stats["Period"] == pd.Timedelta(days=50)
    assert stats["Total Trades"] > 0

    assert len(pf.orders()) > 0
    assert len(pf.trades()) > 0
