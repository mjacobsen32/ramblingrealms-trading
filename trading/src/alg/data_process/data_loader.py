import os
from typing import ClassVar, Dict, Type

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from cli.alg.config import (DataConfig, DataRequests, DataSourceType,
                            FeatureConfig)
from rich import print as rprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.user_cache import user_cache


class DataSource:
    def __init__(self, **kwargs):
        pass

    # Registry for subclasses
    _registry: ClassVar[Dict[DataSourceType, Type["DataSource"]]] = {}

    def get_data(
        self,
        request: DataRequests,
        df: pd.DataFrame,
        cache_path: str,
        start_date: str,
        end_date: str,
        time_step: TimeFrameUnit = TimeFrameUnit.Day,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        type_val = getattr(cls, "TYPE", None)
        if type_val is not None:
            cls._registry[type_val] = cls

    @classmethod
    def factory(cls, data: dict) -> "DataSource":
        datasource_type = data.get("source")
        if not datasource_type:
            raise ValueError("Missing 'type' field in feature data")
        datasource_type_enum = DataSourceType(datasource_type)
        subclass = cls._registry.get(datasource_type_enum)
        if not subclass:
            raise ValueError(f"Unknown DataSourceType type: {datasource_type_enum}")
        return subclass(**data)


class AlpacaDataLoader(DataSource):
    TYPE: ClassVar[DataSourceType] = DataSourceType.ALPACA

    def get_data(
        self,
        request: DataRequests,
        df: pd.DataFrame,
        cache_path: str,
        start_date: str,
        end_date: str,
        time_step: TimeFrameUnit = TimeFrameUnit.Day,
    ) -> pd.DataFrame:
        cache_file = os.path.join(
            cache_path,
            request.dataset_name + ".parquet",
        )
        user = user_cache.UserCache().load()
        if os.path.exists(cache_file):
            rprint("Loading data from cache...")
            df = pd.concat([pd.read_parquet(cache_file), df])
        else:
            rprint("Fetching data from Alpaca...")
            client = StockHistoricalDataClient(
                user.alpaca_api_key.get_secret_value(),
                user.alpaca_api_secret.get_secret_value(),
            )
            request_params = StockBarsRequest(
                timeframe=TimeFrame(1, time_step),
                start=pd.to_datetime(start_date),
                end=pd.to_datetime(end_date),
                **request.kwargs,
            )
            df = pd.concat([df, client.get_stock_bars(request_params).df])
            # df["ticker"] = request.tickers
            df.to_parquet(cache_file)
        return df


class DataLoader:
    def __init__(self, data_config: DataConfig, feature_config: FeatureConfig):
        self.data_config = data_config
        self.feature_config = feature_config
        self.df = pd.DataFrame()
        for request in self.data_config.requests:
            self.df = DataSource.factory(request.model_dump()).get_data(
                request,
                self.df,
                data_config.cache_path,
                data_config.start_date,
                data_config.end_date,
                data_config.time_step,
            )
            rprint(self.df)
            exit(0)

    # ---- FEATURE ENGINEERING ----
    def add_features(self, df):
        df = df.copy()
        # for feature in self.feature_config.features:

        # df["return"] = df["close"].pct_change()
        # df["ma5"] = df["close"].rolling(window=5).mean()
        # df["ma10"] = df["close"].rolling(window=10).mean()
        # df["volatility"] = df["close"].rolling(window=5).std()
        # df["target"] = (df["close"].shift(-1) > df["close"]).astype(
        #     int
        # )  # 1 if next day up, else 0
        # df = df.dropna()
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

    def load_df(self):
        self.df = self.add_trading_signals(self.df, hold_threshold=0.002)
        return self.df

    def get_train_test(self):
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
        X = self.df[features].values
        y = self.df["signal"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
