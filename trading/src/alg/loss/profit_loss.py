import torch
import torch.nn as nn


class ProfitLoss(torch.nn.Module):
    def __init__(self, close_prices, device="cuda"):
        super().__init__()
        self.close_prices = torch.tensor(close_prices, dtype=torch.float32).to(device)

    def forward(self, outputs, targets, indices):
        # outputs: [batch, 3], targets: [batch], indices: [batch]
        probs = torch.softmax(outputs, dim=1)  # [batch, 3]

        batch_close = self.close_prices[indices]
        batch_next_close = self.close_prices[indices + 1]
        change = (batch_next_close - batch_close) / batch_close  # percent change

        # profit = [SELL, HOLD, BUY]
        # SELL = price drops → profit
        # BUY = price rises → profit
        # HOLD = 0
        profit_vector = torch.stack(
            [-change, torch.zeros_like(change), change], dim=1  # SELL  # HOLD  # BUY
        )  # [batch, 3]

        expected_profit = torch.sum(probs * profit_vector, dim=1)  # [batch]
        loss = -expected_profit.mean()  # negative to maximize expected profit
        return loss
