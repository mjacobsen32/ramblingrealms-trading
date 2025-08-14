from enum import Enum
from typing import Any, ClassVar, Dict, List, Type

import numpy as np
import pandas as pd
import vectorbt as vbt
from alpaca.data.timeframe import TimeFrameUnit
from pydantic import BaseModel, Field, model_validator
from scipy.spatial.distance import mahalanobis
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

    def clean_columns(self, cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by filling NaN values in specified columns with given strategy.
        Args:
            cols (List[str]): List of column names to clean.
            df (pd.DataFrame): DataFrame to clean.
        Returns:
            pd.DataFrame: Cleaned DataFrame with NaN values handled according to the fill strategy.
        """

        if self.fill_strategy is FillStrategy.DROP:
            df.dropna(axis=0, inplace=True, subset=cols)
            return df

        for col in cols:
            if self.fill_strategy is FillStrategy.BACKWARD_FILL:
                df[col] = df[col].bfill()
            elif self.fill_strategy is FillStrategy.INTERPOLATE:
                interpolated = df[col].interpolate(
                    method="linear", limit_direction="both"
                )
                df[col] = df[col].where(~df[col].isna(), interpolated)
            elif self.fill_strategy is FillStrategy.ZERO:
                df[col] = df[col].fillna(0.0)

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
        df["open_norm"] = df["open"].pct_change().fillna(0.0)
        df["high_norm"] = df["high"].pct_change().fillna(0.0)
        df["low_norm"] = df["low"].pct_change().fillna(0.0)
        df["close_norm"] = df["close"].pct_change().fillna(0.0)
        df["vwap_norm"] = df["vwap"].pct_change().fillna(0.0)

        # large swings, so we log the percent change, and pipe to z-score
        df["trade_count_norm"] = (
            df["trade_count"]
            .pct_change()
            .fillna(0.0)
            .transform(np.log1p)
            .pipe(Feature._robust_zscore)
        )
        df["volume_norm"] = (
            df["volume"]
            .pct_change()
            .fillna(0.0)
            .transform(np.log1p)
            .pipe(Feature._robust_zscore)
        )

        return df

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        cleaned = self.clean_columns(self.get_raw_feature_names(), data)
        if "symbol" in (df.index.names or []):
            normalized = cleaned.groupby(level="symbol", group_keys=False).apply(
                self.normalize
            )
        else:
            normalized = self.normalize(cleaned)
        return normalized

    TYPE: ClassVar[FeatureType] = FeatureType.CANDLE


class MovingWindow(Feature):
    """
    Represents a moving average feature by Days.
    Note: This feature is readily available in vectorbt, but this implementation
    allows for custom operations and fields.
    """

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        df[self.name] = self.operation(df, self.field, self.period)
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
        df[self.name] = vbt.ATR.run(
            high=df[self.field_high],
            low=df[self.field_low],
            close=df[self.field_close],
            window=self.period,
        ).atr
        df[self.name + "_norm"] = df[self.name] / df["close"]
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
        df[self.name] = vbt.RSI.run(
            close=df[self.field],
            window=self.period,
            ewm=self.ewm,
        ).rsi
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
        macd = vbt.MACD.run(
            close=df[self.field],
            fast_window=self.fast_period,
            slow_window=self.slow_period,
            signal_window=self.signal_period,
            macd_ewm=self.ewm,
            signal_ewm=self.signal_ewm,
        )
        df[f"{self.name}"] = macd.macd
        df[f"{self.name}_signal"] = macd.signal
        df[f"{self.name}_hist"] = macd.macd - macd.signal
        if "symbol" in (df.index.names or []):
            df = df.groupby(level="symbol", group_keys=False).apply(self.normalize)
        else:
            df = self.normalize(df)
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
        bbands = vbt.BBANDS.run(
            close=df[self.field],
            window=self.period,
            ewm=self.ewm,
            alpha=self.alpha,
        )
        df[f"{self.name}_upper"] = bbands.upper
        df[f"{self.name}_lower"] = bbands.lower
        df[f"{self.name}_mid"] = bbands.middle
        if "symbol" in (df.index.names or []):
            df = df.groupby(level="symbol", group_keys=False).apply(self.normalize)
        else:
            df = self.normalize(df)
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
        df[self.name] = vbt.MSTD.run(
            close=df[self.field],
            window=self.window,
            ewm=self.ewm,
        ).mstd
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
        df[self.name] = vbt.OBV.run(
            close=df[self.field_close],
            volume=df[self.field_volume],
        ).obv
        if "symbol" in (df.index.names or []):
            df = df.groupby(level="symbol", group_keys=False).apply(self.normalize)
        else:
            df = self.normalize(df)
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
        kd = vbt.STOCH.run(
            high=df[self.field_high],
            low=df[self.field_low],
            close=df[self.field_close],
            k_window=self.k_window,
            d_window=self.d_window,
            d_ewm=self.ewm,
        )
        df[self.name + "_k"] = kd.percent_k
        df[self.name + "_d"] = kd.percent_d
        df[self.name] = kd.percent_k - kd.percent_d  # Stochastic Oscillator value
        return self.clean_columns(self.get_feature_names(), df)


class Turbulence(Feature):
    """
    Computes a market turbulence index using multiple return periods and proper covariance calculation.
    This measures how unusual current market conditions are compared to historical patterns.
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
