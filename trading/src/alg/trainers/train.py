import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from trading.src.alg.models.hf_time_series import SimpleMLP3
from trading.src.alg.loss.profit_loss import ProfitLoss
import trading.cli.alg.config as rr_config
import trading.src.user_cache.user_cache as rr_user_cache
import trading.src.alg.data_process.data_loader as rr_data_loader
from rich import print as rprint
import vectorbt as vbt


class Trainer:
    def __init__(
        self,
        alg_config: rr_config.AlgConfig,
        user_cache: rr_user_cache.UserCache,
        data_loader: rr_data_loader.DataLoader,
    ):
        self.alg_config = alg_config
        self.user_cache = user_cache

        df = data_loader.load_df()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = ProfitLoss(df["close"].values)

        self.X_train, self.X_test, self.y_train, self.y_test = (
            data_loader.get_train_test()
        )

        self.model = SimpleMLP3(self.X_train.shape[1]).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.batch_size = 16
        self.seq_len = 20
        self.epochs = 5
        self.close_prices = df["close"].values
        torch.manual_seed(42)  # For reproducibility

    def train(self):
        self.model.train()
        n = len(self.X_train)
        for epoch in range(self.epochs):
            total_loss = 0
            batch_count = 0
            permutation = np.random.permutation(
                n - 1
            )  # n-1 to avoid index error with next day's price
            for i in range(
                0, n - self.batch_size, self.batch_size
            ):  # ensure we have enough samples for next day's price
                batch_indices = permutation[i : i + self.batch_size]
                batch_x = torch.tensor(
                    self.X_train[batch_indices], dtype=torch.float32
                ).to(self.device)
                batch_y = torch.tensor(
                    self.y_train[batch_indices], dtype=torch.long
                ).to(self.device)
                batch_indices_tensor = torch.as_tensor(
                    batch_indices, dtype=torch.long, device=self.device
                )

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                # Use the custom profit-based loss function
                loss = self.loss_fn(outputs, batch_y, batch_indices_tensor)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count
            rprint(f"Epoch {epoch+1}/{self.epochs}, Profit-based Loss: {avg_loss:.6f}")

    def test(self):
        self.model.eval()
        actions = ["SELL", "HOLD", "BUY"]
        symbol = "AAPL"  # You can make this dynamic or configurable

        with torch.no_grad():
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(
                self.device
            )
            outputs = self.model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        position = 0  # Track if we're holding a position

        entries = np.zeros_like(predictions, dtype=bool)
        exits = np.zeros_like(predictions, dtype=bool)

        for i, action_id in enumerate(predictions):
            action = actions[action_id]

            # Execute action using Alpaca
            if action == "BUY":
                entries[i] = True
                position += 1
                # rprint(f"BUY executed at ${price:.2f}")
            elif action == "SELL" and position > 0:
                exits[i] = True
                # rprint(f"SELL executed at ${price:.2f}")
                position -= 1
        exits[-1] = True

        entries = pd.Series(entries)
        exits = pd.Series(exits)

        rprint(entries)
        rprint(exits)

        pf = vbt.Portfolio.from_signals(
            close=self.close_prices[-len(predictions)],
            price=self.close_prices[-len(predictions)],
            size=1,
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.001,
        )
        rprint(pf.stats())
        pf.plot().show()
        print("[green]Backtest complete.")
