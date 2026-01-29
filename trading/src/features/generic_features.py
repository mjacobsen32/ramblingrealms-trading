from enum import Enum
from typing import Any, ClassVar, Dict, List, Type

import numpy as np
import pandas as pd
import vectorbt as vbt
from pydantic import BaseModel, Field
from scipy.special import expit


class FeatureType(str, Enum):
    """
    Enum for feature types.
    """

    CANDLE = "candle"
    MOVING_WINDOW = "moving_window"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    STOCHASTIC = "stochastic"
    PIOTROSKI = "piotroski"
    TURBULENCE = "turbulence"
    ATR = "atr"
    MSTD = "mstd"
    OBV = "obv"
    # EDGAR Financial Features
    EDGAR_FUNDAMENTALS = "edgar_fundamentals"
    EDGAR_PROFITABILITY = "edgar_profitability"
    EDGAR_LIQUIDITY = "edgar_liquidity"
    EDGAR_GROWTH = "edgar_growth"
    EDGAR_VALUATION = "edgar_valuation"


class FillStrategy(str, Enum):
    """
    Enum for fill strategies used in Feature cleaning.
    """

    INTERPOLATE = "interpolate"
    ZERO = "zero"
    DROP = "drop"
    BACKWARD_FILL = "bfill"


class OperationType(str, Enum):
    """
    Enum for rolling operations.
    """

    MEAN = "mean"
    STD = "std"

    def __call__(self, df: pd.DataFrame, column: str, window: int) -> pd.Series:
        """
        Apply the rolling operation to a DataFrame column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Name of the column to operate on.
            window (int): Rolling window size.

        Returns:
            pd.Series: Result of the rolling operation.
        """

        if not hasattr(df[column].rolling(window=window), self.value):
            raise ValueError(f"{self.value} is not a valid rolling method")
        return getattr(df[column].rolling(window=window), self.value)()


class Feature(BaseModel):
    """
    Represents a single feature with its name and value.
    This is an abstract base class for all features.
    Subclasses should implement the `to_df` method to convert feature data into a DataFrame.
    It also provides methods for cleaning columns, getting feature names, and a factory method
    to create instances based on a dictionary input.
    The `Feature` class is not meant to be instantiated directly; instead, it serves as a base class
    for specific feature implementations like `Candle`, `MovingWindow`, `RSI`, etc.
    Subclasses must define the `TYPE` class variable to specify their feature type.
    Subclasses can also define additional fields specific to their feature type.
    """

    type: FeatureType
    name: str = Field(..., description="Name of the feature")
    enabled: bool = Field(True, description="Whether the feature is enabled or not")
    source: str | None = Field(None, description="Source file of the feature data")
    fill_strategy: FillStrategy = Field(
        FillStrategy.BACKWARD_FILL, description="Strategy for filling missing values"
    )

    # Registry for subclasses
    _registry: ClassVar[Dict[FeatureType, Type["Feature"]]] = {}

    def shape(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Shape the data to ensure all symbols have the same timestamps.
        Missing timestamps for symbols are filled with NaNs.

        Args:
            data (pd.DataFrame): Input DataFrame with MultiIndex (timestamp, symbol).

        Returns:
            pd.DataFrame: Shaped DataFrame where all symbols have identical timestamps.
        """
        # Get all unique timestamps across all symbols
        all_timestamps = data.index.get_level_values("timestamp").unique()

        # Unstack to have symbols as columns
        unstacked = data.unstack(level="symbol")

        # Reindex to ensure all timestamps are present
        unstacked = unstacked.reindex(all_timestamps)

        # Stack back with dropna=False to keep NaNs
        shaped = unstacked.stack(level="symbol", dropna=False)

        return shaped

    def clean_columns(self, cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by filling NaN values in specified columns with given strategy.
        Filling is applied per symbol to avoid cross-contamination between symbols.
        Args:
            cols (List[str]): List of column names to clean.
            df (pd.DataFrame): DataFrame to clean.
        Returns:
            pd.DataFrame: Cleaned DataFrame with NaN values handled according to the fill strategy.
        """

        if self.fill_strategy is FillStrategy.DROP:
            df.dropna(axis=0, inplace=True, subset=cols)
            return df

        # Apply filling per symbol to prevent using data from other symbols
        grouped = df.groupby(level="symbol", group_keys=False)

        def apply_fill(group):
            for col in cols:
                if self.fill_strategy is FillStrategy.BACKWARD_FILL:
                    group[col] = group[col].bfill()
                elif self.fill_strategy is FillStrategy.INTERPOLATE:
                    interpolated = group[col].interpolate(
                        method="linear", limit_direction="both"
                    )
                    group[col] = group[col].where(~group[col].isna(), interpolated)
                elif self.fill_strategy is FillStrategy.ZERO:
                    group[col] = group[col].fillna(0.0)
                else:
                    raise ValueError(f"Unknown fill strategy: {self.fill_strategy}")
            return group

        df = grouped.apply(apply_fill)
        return df

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features provided by this instance.
        Returns:
            List[str]: List of feature names.
        """
        return [self.name]

    def __init__(self, **data):
        """
        Initialize the Feature instance.
        Args:
            **data: Keyword arguments to initialize the feature.
        Raises:
            TypeError: If the Feature class is instantiated directly.
        Raises:
            TypeError: If the Feature class is instantiated directly.
        """
        if type(self) is Feature:
            raise TypeError(
                "Feature is an abstract base class and cannot be instantiated directly."
            )
        super().__init__(**data)

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        """
        Convert the feature data to a DataFrame.
        Args:
            df (pd.DataFrame): DataFrame to which the feature will be added.
            data (Any): Data to convert into a DataFrame.
        Returns:
            pd.DataFrame: DataFrame with the feature data added.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """

        raise TypeError(
            "Feature is an abstract base class and cannot be instantiated directly."
        )

    def __init_subclass__(cls, **kwargs):
        """
        Register subclasses of Feature in the _registry dictionary.
        """

        super().__init_subclass__(**kwargs)
        type_val = getattr(cls, "TYPE", None)
        if type_val is not None:
            cls._registry[type_val] = cls

    @classmethod
    def factory(cls, data: dict, fill_strategy: FillStrategy) -> "Feature":
        """
        Factory method to create a Feature instance from a dictionary.
        Args:
            data (dict): Dictionary containing feature data, must include 'type'.
        Returns:
            Feature: An instance of a subclass of Feature based on the 'type' field.
        Raises:
            ValueError: If 'type' is missing or if the type is unknown.
        """

        feature_type = data.get("type")
        if not feature_type:
            raise ValueError("Missing 'type' field in feature data")
        feature_type_enum = FeatureType(feature_type)
        subclass = cls._registry.get(feature_type_enum)
        if not subclass:
            raise ValueError(f"Unknown feature type: {feature_type_enum}")
        sub = subclass(**data)
        if sub.fill_strategy is None:
            sub.fill_strategy = fill_strategy
        return sub

    # ---------------------- Normalization Utilities ---------------------- #
    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=s.index)
        return (s - mu) / (std + 1e-12)

    @staticmethod
    def _robust_zscore(s: pd.Series) -> pd.Series:
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0 or np.isnan(mad):
            return pd.Series(0.0, index=s.index)
        return (s - med) / (1.4826 * mad + 1e-12)

    @staticmethod
    def _minmax(s: pd.Series) -> pd.Series:
        mn = s.min()
        mx = s.max()
        if mx - mn == 0 or np.isnan(mx - mn):
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    @staticmethod
    def _sigmoid_pct_change(s: pd.Series) -> pd.Series:
        return expit(s.pct_change().fillna(0.0))


class Candle(Feature):
    """
    Represents a candle feature with OHLCV data.
    """

    def get_feature_names(self) -> List[str]:
        return [
            "open_norm",
            "high_norm",
            "low_norm",
            "close_norm",
            "volume_norm",
            "trade_count_norm",
            "vwap_norm",
        ]

    def get_raw_feature_names(self) -> List[str]:
        return ["open", "high", "low", "close", "volume", "trade_count", "vwap"]

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # percent change from previous day
        # for now we are not piping to z-score, but we may later on, ideally we preserve large swings in raw data
        df["open_norm"] = df["open"].fillna(value=0.0).pct_change().fillna(0.0)
        df["high_norm"] = df["high"].fillna(value=0.0).pct_change().fillna(0.0)
        df["low_norm"] = df["low"].fillna(value=0.0).pct_change().fillna(0.0)
        df["close_norm"] = df["close"].fillna(value=0.0).pct_change().fillna(0.0)
        df["vwap_norm"] = df["vwap"].fillna(value=0.0).pct_change().fillna(0.0)
        df["price"] = df["close"].fillna(0.0)

        # large swings, so we log the percent change, and pipe to z-score
        df["trade_count_norm"] = (
            df["trade_count"]
            .fillna(0.0)
            .pct_change()
            .fillna(0.0)
            .transform(np.log1p)
            .pipe(Feature._robust_zscore)
        )
        df["volume_norm"] = (
            df["volume"]
            .fillna(0.0)
            .pct_change()
            .fillna(0.0)
            .transform(np.log1p)
            .pipe(Feature._robust_zscore)
        )

        return df

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        shaped = self.shape(data)
        cleaned = self.clean_columns(self.get_raw_feature_names(), shaped)
        if "symbol" in cleaned.index.names:
            normalized: pd.DataFrame = cleaned.groupby(
                level="symbol", group_keys=False
            ).apply(self.normalize)
        else:
            normalized = self.normalize(cleaned)
        normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
        normalized.fillna(0.0, inplace=True)
        return normalized

    TYPE: ClassVar[FeatureType] = FeatureType.CANDLE


class MovingWindow(Feature):
    """
    Represents a moving average feature by Days.
    Note: This feature is readily available in vectorbt, but this implementation
    allows for custom operations and fields.
    """

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        grouped = df.groupby(level="symbol", group_keys=False)

        def apply_moving(group):
            group[self.name] = self.operation(group, self.field, self.period)
            return group

        df = grouped.apply(apply_moving)
        return self.clean_columns(self.get_feature_names(), df)

    TYPE: ClassVar[FeatureType] = FeatureType.MOVING_WINDOW
    period: int = Field(
        ...,
        description="The period for the moving operation (e.g., 20 for a 20-day Moving (average))",
    )
    field: str = Field("close", description="Field to roll")
    operation: OperationType = Field(
        OperationType.MEAN, description="Operation to perform on rolling values"
    )


class ATR(Feature):
    """
    Average True Range (ATR) feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.ATR
    period: int = Field(14, description="Period for ATR calculation")
    field_high: str = Field("high", description="Field for high prices")
    field_low: str = Field("low", description="Field for low prices")
    field_close: str = Field("close", description="Field for close prices")

    def get_feature_names(self) -> List[str]:
        return [self.name + "_norm"]

    def get_raw_feature_names(self) -> List[str]:
        return [self.name]

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        grouped = df.groupby(level="symbol", group_keys=False)

        def apply_atr(group):
            group[self.name] = vbt.ATR.run(
                high=group[self.field_high],
                low=group[self.field_low],
                close=group[self.field_close],
                window=self.period,
            ).atr
            group[self.name + "_norm"] = group[self.name] / group["close"]
            return group

        df = grouped.apply(apply_atr)
        cleaned = self.clean_columns([self.name, self.name + "_norm"], df)
        return cleaned


class RSI(Feature):
    """
    Relative Strength Index (RSI) feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.RSI
    period: int = Field(14, description="Period for RSI calculation")
    field: str = Field("close", description="Field to calculate RSI on")
    ewm: bool = Field(False, description="Use Exponential Weighted Average for RSI")

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        grouped = df.groupby(level="symbol", group_keys=False)

        def apply_rsi(group):
            group[self.name] = vbt.RSI.run(
                close=group[self.field],
                window=self.period,
                ewm=self.ewm,
            ).rsi
            return group

        df = grouped.apply(apply_rsi)
        return self.clean_columns(self.get_feature_names(), df)


class MACD(Feature):
    """
    Moving Average Convergence Divergence (MACD) feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.MACD
    fast_period: int = Field(12)
    slow_period: int = Field(26)
    signal_period: int = Field(9)
    field: str = Field("close")
    ewm: bool = Field(
        False, description="Use Exponential Weighted Moving Average for MACD"
    )
    signal_ewm: bool = Field(
        False,
        description="Use Exponential Weighted Moving Average for MACD signal line",
    )

    def get_feature_names(self) -> List[str]:
        return [
            f"{self.name}_norm",
            f"{self.name}_norm_signal",
            f"{self.name}_norm_hist",
        ]

    def get_raw_feature_names(self) -> List[str]:
        return [f"{self.name}", f"{self.name}_signal", f"{self.name}_hist"]

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f"{self.name}_norm"] = self._zscore(df[f"{self.name}"])
        df[f"{self.name}_norm_signal"] = self._zscore(df[f"{self.name}_signal"])
        df[f"{self.name}_norm_hist"] = self._zscore(df[f"{self.name}_hist"])
        return df

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        grouped = df.groupby(level="symbol", group_keys=False)

        def apply_macd(group):
            macd = vbt.MACD.run(
                close=group[self.field],
                fast_window=self.fast_period,
                slow_window=self.slow_period,
                signal_window=self.signal_period,
                macd_ewm=self.ewm,
                signal_ewm=self.signal_ewm,
            )
            group[f"{self.name}"] = macd.macd
            group[f"{self.name}_signal"] = macd.signal
            group[f"{self.name}_hist"] = macd.macd - macd.signal
            return self.normalize(group)

        df = grouped.apply(apply_macd)
        return self.clean_columns(
            self.get_feature_names() + self.get_raw_feature_names(), df
        )


class BollingerBands(Feature):
    """
    Bollinger Bands feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.BOLLINGER_BANDS
    period: int = Field(20)
    field: str = Field("close")
    ewm: bool = Field(False, description="Use Exponential Weighted Moving Average")
    alpha: float = Field(
        2.0, description="Standard deviation multiplier for Bollinger Bands"
    )

    def get_feature_names(self) -> List[str]:
        return [
            f"{self.name}_norm_upper",
            f"{self.name}_norm_lower",
            f"{self.name}_norm_mid",
        ]

    def get_raw_feature_names(self) -> List[str]:
        return [f"{self.name}_upper", f"{self.name}_lower", f"{self.name}_mid"]

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f"{self.name}_norm_upper"] = self._zscore(df[f"{self.name}_upper"])
        df[f"{self.name}_norm_lower"] = self._zscore(df[f"{self.name}_lower"])
        df[f"{self.name}_norm_mid"] = self._zscore(df[f"{self.name}_mid"])
        return df

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        grouped = df.groupby(level="symbol", group_keys=False)

        def apply_bbands(group):
            bbands = vbt.BBANDS.run(
                close=group[self.field],
                window=self.period,
                ewm=self.ewm,
                alpha=self.alpha,
            )
            group[f"{self.name}_upper"] = bbands.upper
            group[f"{self.name}_lower"] = bbands.lower
            group[f"{self.name}_mid"] = bbands.middle
            return self.normalize(group)

        df = grouped.apply(apply_bbands)
        return self.clean_columns(
            self.get_feature_names() + self.get_raw_feature_names(), df
        )


class MSTD(Feature):
    """
    Moving Standard Deviation feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.MSTD
    field: str = Field(
        "close", description="Field to calculate moving standard deviation on"
    )
    window: int = Field(
        20, description="Window size for moving standard deviation calculation"
    )
    ewm: bool = Field(
        False, description="Use Exponential Weighted Moving Standard Deviation"
    )

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        grouped = df.groupby(level="symbol", group_keys=False)

        def apply_mstd(group):
            group[self.name] = vbt.MSTD.run(
                close=group[self.field],
                window=self.window,
                ewm=self.ewm,
            ).mstd
            return group

        df = grouped.apply(apply_mstd)
        return self.clean_columns(self.get_feature_names(), df)


class OBV(Feature):
    """
    On-Balance Volume (OBV) feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.OBV
    field_close: str = Field("close", description="Field for close prices")
    field_volume: str = Field("volume", description="Field for volume")

    def get_feature_names(self) -> List[str]:
        return [f"{self.name}_norm"]

    def get_raw_feature_names(self) -> List[str]:
        return [f"{self.name}"]

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f"{self.name}_norm"] = self._zscore(df[f"{self.name}"])
        return df

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        grouped = df.groupby(level="symbol", group_keys=False)

        def apply_obv(group):
            group[self.name] = vbt.OBV.run(
                close=group[self.field_close],
                volume=group[self.field_volume],
            ).obv
            return self.normalize(group)

        df = grouped.apply(apply_obv)
        return self.clean_columns(self.get_feature_names(), df)


class Stochastic(Feature):
    """
    Stochastic Oscillator feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.STOCHASTIC
    period: int = Field(14)
    field_high: str = Field("high")
    field_low: str = Field("low")
    field_close: str = Field("close")
    k_window: int = Field(14, description="Window for %K smoothing")
    d_window: int = Field(3, description="Window for %D smoothing")
    ewm: bool = Field(
        False, description="Use Exponential Weighted Moving Average for %K and %D"
    )

    def get_feature_names(self) -> List[str]:
        return [f"{self.name}_k", f"{self.name}_d", self.name]

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        grouped = df.groupby(level="symbol", group_keys=False)

        def apply_stoch(group):
            kd = vbt.STOCH.run(
                high=group[self.field_high],
                low=group[self.field_low],
                close=group[self.field_close],
                k_window=self.k_window,
                d_window=self.d_window,
                d_ewm=self.ewm,
            )
            group[self.name + "_k"] = kd.percent_k
            group[self.name + "_d"] = kd.percent_d
            group[self.name] = (
                kd.percent_k - kd.percent_d
            )  # Stochastic Oscillator value
            return group

        df = grouped.apply(apply_stoch)
        return self.clean_columns(self.get_feature_names(), df)


class Turbulence(Feature):
    """
    Computes a market turbulence index using multiple return periods and proper covariance calculation.
    This measures how unusual current market conditions are compared to historical patterns.
    ! TODO pretty sure turbulence is broken
    """

    TYPE: ClassVar[FeatureType] = FeatureType("turbulence")
    lookback: int = Field(20, description="Lookback window for calculating turbulence")
    field: str = Field("close", description="Field to use for return computation")
    return_periods: List[int] = Field(
        [1, 5, 10], description="Return periods to use for multivariate analysis"
    )

    def to_df(self, df: pd.DataFrame, data: Any = None) -> pd.DataFrame:
        # Calculate returns for multiple periods
        returns_matrix = []
        for period in self.return_periods:
            returns = df[self.field].pct_change(periods=period).dropna()
            returns_matrix.append(returns)

        # Align all return series to the same length
        min_length = min(len(r) for r in returns_matrix)
        returns_df = pd.DataFrame(
            {
                f"return_{period}d": r.iloc[-min_length:].values
                for period, r in zip(self.return_periods, returns_matrix)
            }
        )

        turbulence_values = []

        for i in range(len(returns_df)):
            if i < self.lookback:
                turbulence_values.append(np.nan)
                continue

            # Historical window for covariance calculation
            start_idx = max(0, i - self.lookback)
            historical_returns = returns_df.iloc[start_idx:i]
            current_returns = returns_df.iloc[i].values

            # Skip if insufficient data
            if len(historical_returns) < 3:
                turbulence_values.append(0)
                continue

            try:
                # Calculate mean and covariance matrix
                mean_returns = historical_returns.mean().values
                cov_matrix = historical_returns.cov().values

                # Add small regularization to avoid singular matrix
                cov_matrix += np.eye(len(cov_matrix)) * 1e-8

                # Calculate Mahalanobis distance (turbulence index)
                diff = np.array(current_returns) - np.array(mean_returns)
                inv_cov = np.linalg.pinv(cov_matrix)  # Use pseudo-inverse for stability
                turbulence = np.dot(np.dot(diff.T, inv_cov), diff)
                turbulence_values.append(float(turbulence))

            except (np.linalg.LinAlgError, ValueError):
                # Fallback to simple standardized distance if matrix issues
                mean_returns = np.array(historical_returns.mean().values)
                std_returns = np.array(historical_returns.std().values)
                std_returns = np.where(std_returns == 0, 1e-8, std_returns)
                normalized_diff = (
                    np.array(current_returns) - mean_returns
                ) / std_returns
                turbulence_values.append(float(np.sum(normalized_diff**2)))

        # Align turbulence values with original dataframe
        result_values = [np.nan] * (
            len(df) - len(turbulence_values)
        ) + turbulence_values
        df[self.name] = result_values

        return self.clean_columns(self.get_feature_names(), df)


# ======================== EDGAR Financial Features ======================== #


class EdgarFundamentals(Feature):
    """
    EDGAR Fundamentals feature - extracts core financial statement data.

    This feature provides access to fundamental financial data from SEC EDGAR filings:
    - Revenue and revenue-related metrics
    - Net income and earnings
    - Assets, liabilities, and equity
    - Cash flow components

    The data is aligned by filing date (when it was released) to avoid look-ahead bias.
    Data is forward-filled from each filing until the next one is released.

    Note: This feature requires the EdgarDataLoader to be run first to populate
    the edgar_* columns in the dataframe.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.EDGAR_FUNDAMENTALS

    # Which fundamental metrics to include
    include_revenue: bool = Field(True, description="Include revenue metrics")
    include_income: bool = Field(True, description="Include income metrics")
    include_balance_sheet: bool = Field(
        True, description="Include balance sheet metrics"
    )
    include_cash_flow: bool = Field(True, description="Include cash flow metrics")
    normalize: bool = Field(True, description="Normalize values using log transform")

    def get_feature_names(self) -> List[str]:
        """Get the feature names based on configuration."""
        features = []

        if self.include_revenue:
            features.extend(
                [
                    f"{self.name}_revenue",
                    f"{self.name}_gross_profit",
                ]
            )

        if self.include_income:
            features.extend(
                [
                    f"{self.name}_net_income",
                    f"{self.name}_operating_income",
                    f"{self.name}_ebitda",
                ]
            )

        if self.include_balance_sheet:
            features.extend(
                [
                    f"{self.name}_total_assets",
                    f"{self.name}_total_liabilities",
                    f"{self.name}_total_equity",
                    f"{self.name}_cash",
                    f"{self.name}_debt",
                ]
            )

        if self.include_cash_flow:
            features.extend(
                [
                    f"{self.name}_operating_cf",
                    f"{self.name}_free_cf",
                ]
            )

        return features

    def get_raw_feature_names(self) -> List[str]:
        return self.get_feature_names()

    def _safe_log_transform(self, series: pd.Series) -> pd.Series:
        """Apply log transform handling negative values and zeros."""
        # Use sign-preserving log transform: sign(x) * log(|x| + 1)
        return np.sign(series) * np.log1p(np.abs(series))

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        """
        Extract EDGAR fundamental features from the dataframe.

        The EdgarDataLoader should have already populated the edgar_* columns.
        This feature class transforms and normalizes them for use in the model.
        """
        if df.empty:
            return df

        # Map edgar columns to our feature names
        column_mapping = {}

        if self.include_revenue:
            column_mapping[f"{self.name}_revenue"] = "edgar_revenue"
            column_mapping[f"{self.name}_gross_profit"] = "edgar_gross_profit"

        if self.include_income:
            column_mapping[f"{self.name}_net_income"] = "edgar_net_income"
            column_mapping[f"{self.name}_operating_income"] = "edgar_operating_income"
            column_mapping[f"{self.name}_ebitda"] = "edgar_ebitda"

        if self.include_balance_sheet:
            column_mapping[f"{self.name}_total_assets"] = "edgar_total_assets"
            column_mapping[f"{self.name}_total_liabilities"] = "edgar_total_liabilities"
            column_mapping[f"{self.name}_total_equity"] = "edgar_total_equity"
            column_mapping[f"{self.name}_cash"] = "edgar_cash_and_equivalents"
            column_mapping[f"{self.name}_debt"] = "edgar_total_debt"

        if self.include_cash_flow:
            column_mapping[f"{self.name}_operating_cf"] = "edgar_operating_cash_flow"
            column_mapping[f"{self.name}_free_cf"] = "edgar_free_cash_flow"

        # Copy and transform columns
        for new_col, edgar_col in column_mapping.items():
            if edgar_col in df.columns:
                if self.normalize:
                    df[new_col] = self._safe_log_transform(df[edgar_col])
                else:
                    df[new_col] = df[edgar_col]
            else:
                df[new_col] = np.nan

        return self.clean_columns(self.get_feature_names(), df)


class EdgarProfitability(Feature):
    """
    EDGAR Profitability feature - extracts profitability ratios and margins.

    This feature provides profitability metrics:
    - Profit margin (Net Income / Revenue)
    - Gross margin (Gross Profit / Revenue)
    - Operating margin (Operating Income / Revenue)
    - Return on Equity (ROE)
    - Return on Assets (ROA)

    These ratios are already normalized (percentages) so no additional
    normalization is typically needed.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.EDGAR_PROFITABILITY

    include_margins: bool = Field(True, description="Include margin ratios")
    include_returns: bool = Field(True, description="Include ROE/ROA")
    clip_outliers: bool = Field(True, description="Clip extreme values")
    clip_range: tuple = Field(
        (-2.0, 2.0), description="Range for clipping (-200% to 200%)"
    )

    def get_feature_names(self) -> List[str]:
        features = []

        if self.include_margins:
            features.extend(
                [
                    f"{self.name}_profit_margin",
                    f"{self.name}_gross_margin",
                    f"{self.name}_operating_margin",
                ]
            )

        if self.include_returns:
            features.extend(
                [
                    f"{self.name}_roe",
                    f"{self.name}_roa",
                ]
            )

        return features

    def get_raw_feature_names(self) -> List[str]:
        return self.get_feature_names()

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        """Extract EDGAR profitability features."""
        if df.empty:
            return df

        column_mapping = {}

        if self.include_margins:
            column_mapping[f"{self.name}_profit_margin"] = "edgar_profit_margin"
            column_mapping[f"{self.name}_gross_margin"] = "edgar_gross_margin"
            column_mapping[f"{self.name}_operating_margin"] = "edgar_operating_margin"

        if self.include_returns:
            column_mapping[f"{self.name}_roe"] = "edgar_roe"
            column_mapping[f"{self.name}_roa"] = "edgar_roa"

        for new_col, edgar_col in column_mapping.items():
            if edgar_col in df.columns:
                values = df[edgar_col].copy()
                if self.clip_outliers:
                    values = values.clip(
                        lower=self.clip_range[0], upper=self.clip_range[1]
                    )
                df[new_col] = values
            else:
                df[new_col] = np.nan

        return self.clean_columns(self.get_feature_names(), df)


class EdgarLiquidity(Feature):
    """
    EDGAR Liquidity feature - extracts liquidity and leverage ratios.

    This feature provides:
    - Current ratio (Current Assets / Current Liabilities)
    - Debt to Equity ratio
    - Cash ratio variations

    These ratios help assess the company's financial health and risk.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.EDGAR_LIQUIDITY

    include_current_ratio: bool = Field(True, description="Include current ratio")
    include_leverage: bool = Field(True, description="Include leverage ratios")
    normalize: bool = Field(True, description="Apply log transform to ratios")

    def get_feature_names(self) -> List[str]:
        features = []

        if self.include_current_ratio:
            features.append(f"{self.name}_current_ratio")

        if self.include_leverage:
            features.append(f"{self.name}_debt_to_equity")

        return features

    def get_raw_feature_names(self) -> List[str]:
        return self.get_feature_names()

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        """Extract EDGAR liquidity features."""
        if df.empty:
            return df

        if self.include_current_ratio:
            col_name = f"{self.name}_current_ratio"
            if "edgar_current_ratio" in df.columns:
                values = df["edgar_current_ratio"].copy()
                if self.normalize:
                    # Log transform: values > 1 are positive, < 1 are negative
                    values = np.log(values.clip(lower=0.01))
                df[col_name] = values
            else:
                df[col_name] = np.nan

        if self.include_leverage:
            col_name = f"{self.name}_debt_to_equity"
            if "edgar_debt_to_equity" in df.columns:
                values = df["edgar_debt_to_equity"].copy()
                if self.normalize:
                    # Sign-preserving log transform
                    values = np.sign(values) * np.log1p(np.abs(values))
                df[col_name] = values
            else:
                df[col_name] = np.nan

        return self.clean_columns(self.get_feature_names(), df)


class EdgarGrowth(Feature):
    """
    EDGAR Growth feature - extracts year-over-year growth metrics.

    This feature provides growth rates:
    - Revenue growth (YoY)
    - Earnings growth (YoY)
    - EPS growth

    Growth rates are useful for identifying companies with momentum.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.EDGAR_GROWTH

    include_revenue_growth: bool = Field(True, description="Include revenue growth")
    include_earnings_growth: bool = Field(True, description="Include earnings growth")
    clip_outliers: bool = Field(True, description="Clip extreme growth values")
    clip_range: tuple = Field(
        (-5.0, 5.0), description="Range for clipping (-500% to 500%)"
    )

    def get_feature_names(self) -> List[str]:
        features = []

        if self.include_revenue_growth:
            features.append(f"{self.name}_revenue_growth")

        if self.include_earnings_growth:
            features.append(f"{self.name}_earnings_growth")

        return features

    def get_raw_feature_names(self) -> List[str]:
        return self.get_feature_names()

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        """Extract EDGAR growth features."""
        if df.empty:
            return df

        if self.include_revenue_growth:
            col_name = f"{self.name}_revenue_growth"
            if "edgar_revenue_growth_yoy" in df.columns:
                values = df["edgar_revenue_growth_yoy"].copy()
                if self.clip_outliers:
                    values = values.clip(
                        lower=self.clip_range[0], upper=self.clip_range[1]
                    )
                df[col_name] = values
            else:
                df[col_name] = np.nan

        if self.include_earnings_growth:
            col_name = f"{self.name}_earnings_growth"
            if "edgar_earnings_growth_yoy" in df.columns:
                values = df["edgar_earnings_growth_yoy"].copy()
                if self.clip_outliers:
                    values = values.clip(
                        lower=self.clip_range[0], upper=self.clip_range[1]
                    )
                df[col_name] = values
            else:
                df[col_name] = np.nan

        return self.clean_columns(self.get_feature_names(), df)


class EdgarValuation(Feature):
    """
    EDGAR Valuation feature - extracts valuation-related metrics.

    This feature provides:
    - EPS (Earnings Per Share)
    - Book Value Per Share
    - Shares Outstanding (normalized)

    Note: P/E ratio and other price-based valuations should be calculated
    by combining these with price data in a separate feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.EDGAR_VALUATION

    include_eps: bool = Field(True, description="Include EPS")
    include_book_value: bool = Field(True, description="Include book value per share")
    include_shares: bool = Field(False, description="Include shares outstanding")
    normalize: bool = Field(True, description="Apply normalization")

    def get_feature_names(self) -> List[str]:
        features = []

        if self.include_eps:
            features.append(f"{self.name}_eps")

        if self.include_book_value:
            features.append(f"{self.name}_book_value")

        if self.include_shares:
            features.append(f"{self.name}_shares")

        return features

    def get_raw_feature_names(self) -> List[str]:
        return self.get_feature_names()

    def _safe_log_transform(self, series: pd.Series) -> pd.Series:
        """Apply log transform handling negative values and zeros."""
        return np.sign(series) * np.log1p(np.abs(series))

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        """Extract EDGAR valuation features."""
        if df.empty:
            return df

        if self.include_eps:
            col_name = f"{self.name}_eps"
            if "edgar_eps" in df.columns:
                values = df["edgar_eps"].copy()
                if self.normalize:
                    values = self._safe_log_transform(values)
                df[col_name] = values
            else:
                df[col_name] = np.nan

        if self.include_book_value:
            col_name = f"{self.name}_book_value"
            if "edgar_book_value_per_share" in df.columns:
                values = df["edgar_book_value_per_share"].copy()
                if self.normalize:
                    values = self._safe_log_transform(values)
                df[col_name] = values
            else:
                df[col_name] = np.nan

        if self.include_shares:
            col_name = f"{self.name}_shares"
            if "edgar_shares_outstanding" in df.columns:
                values = df["edgar_shares_outstanding"].copy()
                if self.normalize:
                    values = np.log1p(values)  # Always positive
                df[col_name] = values
            else:
                df[col_name] = np.nan

        return self.clean_columns(self.get_feature_names(), df)
