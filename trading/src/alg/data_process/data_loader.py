import os
import pandas as pd
import numpy as np
from rich import print as rprint
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from trading.src.user_cache import user_cache
from trading.cli.alg.config import DataConfig, FeatureConfig


class DataLoader:
    def __init__(self, data_config: DataConfig, feature_config: FeatureConfig):
        self.data_config = data_config
        self.feature_config = feature_config

    # ---- FEATURE ENGINEERING ----
    def add_features(self, df):
        df = df.copy()
        df["return"] = df["close"].pct_change()
        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["volatility"] = df["close"].rolling(window=5).std()
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(
            int
        )  # 1 if next day up, else 0
        df = df.dropna()
        return df

    # 0 = SELL, 1 = HOLD, 2 = BUY
    def add_trading_signals(self, df, hold_threshold=0.002):
        df = df.copy()
        df["future_return"] = df["close"].shift(-1) / df["close"] - 1
        # BUY if next day's return > hold_threshold, SELL if < -hold_threshold, else HOLD
        df["signal"] = np.where(
            df["future_return"] > hold_threshold,
            2,
            np.where(df["future_return"] < -hold_threshold, 0, 1),
        )
        df = df.dropna()
        return df

    def load_tensors(self):
        rprint("Loading data tensors...")
        df = self.fetch_and_cache_data()
        df = self.add_trading_signals(df, hold_threshold=0.002)
        df = self.add_features(df)
        return df

    def prepare_data(self, df):
        features = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ma5",
            "ma10",
            "volatility",
        ]
        X = df[features].values
        y = df["signal"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    def fetch_and_cache_data(self):
        cache_file = os.path.join(
            self.data_config.cache_path, self.data_config.dataset_name
        )
        user = user_cache.UserCache().load()
        if os.path.exists(cache_file):
            rprint("Loading data from cache...")
            df = pd.read_parquet(cache_file)
        else:
            rprint("Fetching data from Alpaca...")
            client = StockHistoricalDataClient(
                user.alpaca_api_key.get_secret_value(),
                user.alpaca_api_secret.get_secret_value(),
            )
            all_data = []
            for ticker in self.data_config.tickers:
                request_params = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Day,
                    start=pd.to_datetime(self.data_config.start_date),
                    end=pd.to_datetime(self.data_config.end_date),
                )
                bars = client.get_stock_bars(request_params).df
                bars["ticker"] = ticker
                all_data.append(bars)
            df = pd.concat(all_data)
            df.to_parquet(cache_file)
        return df
