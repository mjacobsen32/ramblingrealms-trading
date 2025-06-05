import backtrader as bt
import torch
import numpy as np


class MLStrategy(bt.Strategy):
    def __init__(self, model):
        self.model = model
        self.dataclose = self.datas[0].close

    def run(self):
        cerebro = bt.Cerebro()
        datafeed = bt.feeds.YahooFinanceCSVData(dataname="data/ohlcv.csv")
        cerebro.adddata(datafeed)
        cerebro.addstrategy(MLStrategy, model=self.model)
        cerebro.run()
        cerebro.plot()

    def next(self):
        if len(self) < 20:  # Using fixed seq_len=20 here
            return
        # Prepare model input with latest sequence
        x = (
            torch.tensor(self.dataclose.get(size=20), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        output = self.model(x).detach().numpy()
        action = np.argmax(output)
        # Execute action
        if action == 2:  # BUY
            self.buy()
        elif action == 0:  # SELL
            self.sell()
