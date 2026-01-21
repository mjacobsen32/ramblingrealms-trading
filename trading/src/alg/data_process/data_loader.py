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
        **kwargs,
    ) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement get_data method")

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
        Raises ValueError if the 'source' key is missing or if the type is unknown.
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
    """
    ! TODO add columns to and from parquet

    """

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
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetches data from Alpaca and caches it locally.
        """
        alpaca_history_client: StockHistoricalDataClient | None = kwargs.get(
            "alpaca_history_client", None
        )
        alpaca_trading_client: TradingClient | None = kwargs.get(
            "alpaca_trading_client", None
        )

        if alpaca_history_client is None and fetch_data:
            user = user_cache.UserCache().load()
            alpaca_history_client = StockHistoricalDataClient(
                user.alpaca_api_key.get_secret_value(),
                user.alpaca_api_secret.get_secret_value(),
            )
        if alpaca_trading_client is None and fetch_data:
            user = user_cache.UserCache().load()
            alpaca_trading_client = TradingClient(
                user.alpaca_api_key.get_secret_value(),
                user.alpaca_api_secret.get_secret_value(),
            )

        cache_file = os.path.join(cache_path, request.dataset_name + ".parquet")

        def _coerce_ts(value, tz):
            ts = pd.to_datetime(value)
            if tz is None:
                return ts.tz_localize(None) if ts.tzinfo else ts
            if ts.tzinfo is None:
                return ts.tz_localize(tz)
            return ts.tz_convert(tz)

        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)

        cache_df = pd.DataFrame()
        used_cache = False
        cache_coverage = None

        if cache_enabled and os.path.exists(cache_file) and not fetch_data:
            cache_df = pd.read_parquet(cache_file)
            used_cache = True
            cache_ts_index = cache_df.index.get_level_values("timestamp")
            cache_tz = getattr(cache_ts_index, "tz", None)
            cache_start = _coerce_ts(cache_ts_index.min(), cache_tz)
            cache_end = _coerce_ts(cache_ts_index.max(), cache_tz)
            requested_start = _coerce_ts(requested_start, cache_tz)
            requested_end = _coerce_ts(requested_end, cache_tz)
            cache_coverage = (cache_start, cache_end)
            logging.info(
                "Loaded cache %s covering %s to %s (%d rows)",
                cache_file,
                cache_start,
                cache_end,
                len(cache_df),
            )
        elif not cache_enabled:
            logging.info(
                "Cache disabled; will fetch requested range %s to %s.",
                requested_start,
                requested_end,
            )
        elif fetch_data:
            logging.info(
                "Forced fetch requested; ignoring any existing cache for %s.",
                request.dataset_name,
            )
        else:
            logging.info(
                "Cache missing; will fetch requested range %s to %s.",
                requested_start,
                requested_end,
            )

        if request.kwargs.get("symbol_or_symbols") == ["ALL"]:
            if alpaca_trading_client is None:
                raise ValueError(
                    "alpaca_trading_client must be provided when symbol_or_symbols is ['ALL']"
                )
            all_assets = alpaca_trading_client.get_all_assets(
                GetAssetsRequest(
                    status=AssetStatus.ACTIVE,
                    asset_class=AssetClass.US_EQUITY,
                )
            )
            # Type narrow: get_all_assets returns list of Asset objects
            assets: list[str] = [
                asset.symbol  # type: ignore[union-attr]
                for asset in all_assets
                if hasattr(asset, "tradable") and asset.tradable  # type: ignore[union-attr]
            ]
            request.kwargs["symbol_or_symbols"] = assets
        assets = request.kwargs["symbol_or_symbols"]
        logging.info("Total number of symbols to consider: %d", len(assets))

        segments_to_fetch = []
        if not fetch_data and alpaca_history_client is None:
            # If we're not allowed to fetch and no client provided, we can only use what's in cache
            if not used_cache:
                logging.warning(
                    "No cache available and fetch_data=False with no client provided. Returning empty dataframe."
                )
        elif not used_cache:
            # No cache - fetch everything if we can
            segments_to_fetch.append((requested_start, requested_end))
        else:
            # We have cache - check if we need to fetch missing ranges
            if cache_coverage is None:
                raise ValueError("cache_coverage should not be None when using cache")
            cache_start, cache_end = cache_coverage
            if requested_start < cache_start:
                segments_to_fetch.append((requested_start, cache_start))
            if requested_end > cache_end:
                segments_to_fetch.append((cache_end, requested_end))
            if not segments_to_fetch:
                logging.info(
                    "Cache fully covers requested range %s to %s; no additional fetch needed.",
                    requested_start,
                    requested_end,
                )

        fetched_parts: list[pd.DataFrame] = []
        for seg_start, seg_end in segments_to_fetch:
            logging.info(
                "Fetching missing data for %s to %s (segment length %s).",
                seg_start,
                seg_end,
                seg_end - seg_start,
            )
            for start_idx in range(0, len(assets), 200):
                end_idx = min(start_idx + 200, len(assets))
                logging.info(
                    "Fetching symbols slice %d-%d for %s to %s...",
                    start_idx,
                    end_idx,
                    seg_start,
                    seg_end,
                )
                request.kwargs["symbol_or_symbols"] = assets[start_idx:end_idx]
                request_params = StockBarsRequest(
                    timeframe=TimeFrame(time_step_period, time_step_unit),
                    start=seg_start,
                    end=seg_end,
                    **request.kwargs,
                )
                if alpaca_history_client is None:
                    raise ValueError(
                        "alpaca_history_client must be provided when fetching data"
                    )
                bars = alpaca_history_client.get_stock_bars(request_params)
                if hasattr(bars, "df"):
                    fetched_parts.append(bars.df)
                elif isinstance(bars, dict) and "df" in bars:
                    fetched_parts.append(bars["df"])
                else:
                    try:
                        fetched_parts.append(pd.DataFrame(bars))
                    except Exception:
                        logging.error(
                            "Unknown bars return type from Alpaca client: %s",
                            type(bars),
                        )
                        raise

        new_data = pd.concat(fetched_parts) if fetched_parts else pd.DataFrame()
        if not new_data.empty:
            logging.info(
                "Fetched %d new rows spanning %s to %s for %d symbols.",
                len(new_data),
                new_data.index.get_level_values("timestamp").min(),
                new_data.index.get_level_values("timestamp").max(),
                new_data.index.get_level_values("symbol").nunique(),
            )
        elif segments_to_fetch:
            logging.info("No rows returned for requested missing segments.")

        dataset_df = pd.concat([cache_df, new_data]) if used_cache else new_data
        if not dataset_df.empty:
            dataset_df.sort_index(level=["timestamp", "symbol"], inplace=True)
            dataset_df = dataset_df[~dataset_df.index.duplicated(keep="last")]

        request.kwargs["symbol_or_symbols"] = assets

        if cache_enabled:
            if dataset_df.empty:
                logging.info("Cache not updated because dataset is empty.")
            elif not new_data.empty or not used_cache:
                dataset_df.to_parquet(cache_file)
                logging.info(
                    "Cache saved to %s covering %s to %s (%d rows).",
                    cache_file,
                    dataset_df.index.get_level_values("timestamp").min(),
                    dataset_df.index.get_level_values("timestamp").max(),
                    len(dataset_df),
                )
            else:
                logging.info("Cache already up-to-date; no new file written.")
        else:
            logging.info("Caching is disabled; not saving data to cache.")

        full_df = pd.concat([df, dataset_df]) if not dataset_df.empty else df
        if not full_df.empty:
            full_df.sort_index(level=["timestamp", "symbol"], inplace=True)
            full_df = full_df[~full_df.index.duplicated(keep="last")]
            ts_index = full_df.index.get_level_values("timestamp")
            target_tz = getattr(ts_index, "tz", None)
            req_start = _coerce_ts(requested_start, target_tz)
            req_end = _coerce_ts(requested_end, target_tz)
            filtered = full_df.loc[(ts_index >= req_start) & (ts_index <= req_end)]
        else:
            filtered = full_df

        if not filtered.empty:
            logging.info(
                "Returning data spanning %s to %s (%d rows, %d symbols).",
                filtered.index.get_level_values("timestamp").min(),
                filtered.index.get_level_values("timestamp").max(),
                len(filtered),
                filtered.index.get_level_values("symbol").nunique(),
            )
        else:
            logging.info(
                "No data available for requested range %s to %s.",
                requested_start,
                requested_end,
            )

        return filtered


class DataLoader:
    """
    DataLoader class for loading and processing data from various sources.
    It initializes with a DataConfig and FeatureConfig, fetches data from the specified sources,
    and applies the specified features to the data.

    ! todo no drop on on live
    """

    def __init__(
        self,
        data_config: DataConfig,
        feature_config: FeatureConfig,
        fetch_data: bool = False,
        **kwargs,
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
                **kwargs,
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

        unique_timestamps = sorted(self.df.index.get_level_values("timestamp").unique())
        split_idx = int(
            len(unique_timestamps) * (1 - self.data_config.validation_split)
        )
        train_timestamps = unique_timestamps[:split_idx]
        test_timestamps = unique_timestamps[split_idx:]
        train = self.df[
            self.df.index.get_level_values("timestamp").isin(train_timestamps)
        ]
        test = self.df[
            self.df.index.get_level_values("timestamp").isin(test_timestamps)
        ]
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
