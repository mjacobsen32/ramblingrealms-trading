import os
from typing import ClassVar, Dict, Type

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from rich import print as rprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from trading.cli.alg.config import (
    DataConfig,
    DataRequests,
    DataSourceType,
    FeatureConfig,
)
from trading.src.user_cache import user_cache


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
            df.to_parquet(cache_file)
        return df


class DataLoader:
    def __init__(self, data_config: DataConfig, feature_config: FeatureConfig):
        self.data_config = data_config
        self.feature_config = feature_config

        self.df = pd.DataFrame()
        self.features = pd.Series()
        self.targets = pd.Series()

        for request in self.data_config.requests:
            data = DataSource.factory(request.model_dump()).get_data(
                request,
                self.df,
                data_config.cache_path,
                data_config.start_date,
                data_config.end_date,
                data_config.time_step,
            )
            for feature in [
                f
                for f in self.feature_config.features
                if f.source == request.dataset_name
            ]:
                self.df = feature.to_df(self.df, data)

        self.features = self.df.columns
        self.df.sort_index(level=["timestamp", "symbol"], inplace=True)
        self.df = self.df.reset_index()  # Moves MultiIndex levels to columns
        self.df = self.df.rename(columns={"symbol": "tic"})
        self.df.dropna()

        rprint(
            f"[green]Data Successfully loaded...\n[white]Current features: {[f for f in self.features]}"
        )

    def get_train_test(self):
        split_idx = int(len(self.df) * (1 - self.data_config.validation_split))
        return self.df.iloc[:split_idx], self.df.iloc[split_idx:]
