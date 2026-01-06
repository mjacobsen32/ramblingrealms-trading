import numpy as np
import pytest

from trading.src.portfolio.position import PositionManager
from trading.test.alg.test_fixtures import *


def test_positions(multi_data_loader):
    data = multi_data_loader.get_train_test()[0]
    data["size"] = np.random.choice([-1, 0, 1], size=len(data))
    p = PositionManager(symbols=data.index.get_level_values("symbol").unique().tolist())
    data["timestamp"] = data.index.get_level_values("timestamp")
    data["price"] = data["close"]
    for unique_date in data.index.get_level_values("timestamp").unique():
        df, profit, orders = p.step(df=data.loc[unique_date])
