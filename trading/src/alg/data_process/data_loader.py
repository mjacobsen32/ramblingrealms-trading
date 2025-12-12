import logging
import os
from typing import ClassVar, Dict, Type

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading import Asset, GetAssetsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus

from trading.cli.alg.config import (
    DataConfig,
    DataRequests,
    DataSourceType,
    FeatureConfig,
)
from trading.src.features.utils import get_feature_cols
from trading.src.user_cache import user_cache


class DataSource:
    """
    Base class for all data sources in the Rambling Realms trading system.
    This class provides a common interface for fetching and processing data from various sources.
    Subclasses should implement the `get_data` method to fetch data from their respective sources.
    """

    def __init__(self, **kwargs):
        pass

    # Registry for subclasses
    _registry: ClassVar[Dict[DataSourceType, Type["DataSource"]]] = {}

    def get_data(
        self,
        fetch_data: bool,
        request: DataRequests,
        df: pd.DataFrame,
        cache_path: str,
        start_date: str,
        end_date: str,
        time_step_unit: TimeFrameUnit = TimeFrameUnit("Day"),
        cache_enabled: bool = True,
        time_step_period: int = 1,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        type_val = getattr(cls, "TYPE", None)
        if type_val is not None:
            cls._registry[type_val] = cls

    @classmethod
    def factory(cls, data: dict) -> "DataSource":
        """
        Factory method to create an instance of a DataSource subclass based on the provided data.
        The data dictionary must contain a 'source' key that matches one of the DataSourceType enum values.
        Raises ValueError if the 'source' key is missing or if the type is
        """

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
        fetch_data: bool,
        request: DataRequests,
        df: pd.DataFrame,
        cache_path: str,
        start_date: str,
        end_date: str,
        time_step_unit: str = TimeFrameUnit.Day,
        cache_enabled: bool = True,
        time_step_period: int = 1,
    ) -> pd.DataFrame:
        """
        Fetches data from Alpaca and caches it locally.
        """

        cache_file = os.path.join(
            cache_path,
            request.dataset_name + ".parquet",
        )
        user = user_cache.UserCache().load()
        if os.path.exists(cache_file) and cache_enabled and not fetch_data:
            logging.info("Loading data from cache...")
            df = pd.concat([pd.read_parquet(cache_file), df])
        else:
            logging.info("Fetching data from Alpaca...")
            client = StockHistoricalDataClient(
                user.alpaca_api_key.get_secret_value(),
                user.alpaca_api_secret.get_secret_value(),
            )
            if request.kwargs.get("symbol_or_symbols") == ["ALL"]:
                trading_client = TradingClient(
                    user.alpaca_api_key.get_secret_value(),
                    user.alpaca_api_secret.get_secret_value(),
                )
                assets: list[str] = [
                    asset.symbol
                    for asset in trading_client.get_all_assets(
                        GetAssetsRequest(
                            status=AssetStatus.ACTIVE,
                            asset_class=AssetClass.US_EQUITY,
                        )
                    )
                    if asset.tradable
                ]
                request.kwargs["symbol_or_symbols"] = assets
            assets = request.kwargs["symbol_or_symbols"]
            logging.info("Total number of symbols to fetch: %d", len(assets))
            for start in range(0, len(assets), 200):
                end = min(start + 200, len(assets))
                logging.info("Fetching data for symbols %d to %d...", start, end)
                request.kwargs["symbol_or_symbols"] = assets[start:end]
                request_params = StockBarsRequest(
                    timeframe=TimeFrame(time_step_period, time_step_unit),
                    start=pd.to_datetime(start_date),
                    end=pd.to_datetime(end_date),
                    **request.kwargs,
                )
                bars = client.get_stock_bars(request_params)
                # `get_stock_bars` may return an object with a `.df` attribute or a dict
                if hasattr(bars, "df"):
                    df = pd.concat([df, bars.df])
                elif isinstance(bars, dict) and "df" in bars:
                    df = pd.concat([df, bars["df"]])
                else:
                    # Fallback: try to coerce the return to a DataFrame
                    try:
                        df = pd.concat([df, pd.DataFrame(bars)])
                    except Exception:
                        logging.error(
                            "Unknown bars return type from Alpaca client: %s",
                            type(bars),
                        )
                        raise
            if cache_enabled:
                logging.info("Caching data to %s", cache_file)
                df.to_parquet(cache_file)
            else:
                logging.info("Caching is disabled; not saving data to cache.")

        logging.info("Length of fetched data: %d", len(df))

        logging.info(
            "Processed %d symbols.",
            df.index.get_level_values("symbol").nunique(),
        )

        logging.info(
            "Start of fetched data: %s; End of fetched data: %s",
            df.index.get_level_values("timestamp")[0],
            df.index.get_level_values("timestamp")[-1],
        )
        return df


class DataLoader:
    """
    DataLoader class for loading and processing data from various sources.
    It initializes with a DataConfig and FeatureConfig, fetches data from the specified sources,
    and applies the specified features to the data.
    """

    def __init__(
        self,
        data_config: DataConfig,
        feature_config: FeatureConfig,
        fetch_data: bool = False,
    ):
        """
        Initializes the DataLoader with the given configurations.
        """
        self.data_config = data_config
        self.feature_config = feature_config

        self.df = pd.DataFrame()

        for request in self.data_config.requests:
            data = DataSource.factory(request.model_dump()).get_data(
                fetch_data,
                request,
                self.df,
                str(data_config.cache_path),
                data_config.start_date,
                data_config.end_date,
                TimeFrameUnit(data_config.time_step_unit),
                data_config.cache_enabled,
                data_config.time_step_period,
            )
            for feature in [f for f in self.feature_config.features]:
                if feature.source == request.dataset_name or feature.source is None:
                    self.df = feature.to_df(self.df, data)
                else:
                    logging.warning(
                        "Skipping feature %s for source %s",
                        feature.name,
                        request.dataset_name,
                    )

        # ! if df is empty, raise error probably due to features having mis matched source names

        self.columns = self.df.columns
        self.features = get_feature_cols(features=self.feature_config.features)
        self.df.sort_index(level=["timestamp", "symbol"], inplace=True)
        self.df = self.df.reorder_levels(["timestamp", "symbol"])
        self.df.dropna()

        logging.info(
            "Data Successfully Loaded. Num Rows: %d, Num features: %d",
            len(self.df),
            len(self.features),
        )
        logging.debug(
            "\nCurrent columns: %s\nCurrent Features: %s",
            [f for f in self.columns],
            self.feature_config.features,
        )
        logging.debug(
            "Current tickers: %s",
            self.df.index.get_level_values("symbol").unique().tolist(),
        )

    @classmethod
    def data_info(cls, df: pd.DataFrame) -> str:
        """
        Returns a string representation of the dataframe.
        """
        return f"start_date: {df.index.get_level_values('timestamp').min()}, end_date: {df.index.get_level_values('timestamp').max()}, shape: {df.shape}"

    def get_train_test(self):
        """
        Splits the DataFrame into training and validation sets based on the validation split ratio.
        """

        split_idx = int(len(self.df) * (1 - self.data_config.validation_split))
        train = self.df.iloc[:split_idx]
        test = self.df.iloc[split_idx:]
        logging.info(
            "Train data: %s\nTest data: %s", self.data_info(train), self.data_info(test)
        )

        return train, test

    def to_csv(self, path: str):
        """
        Saves the DataFrame to a CSV file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.df.to_csv(
            path,
            index=True,
        )
        logging.info("Data saved to %s", path)
