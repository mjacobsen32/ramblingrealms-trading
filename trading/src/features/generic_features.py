from enum import Enum
from typing import Any, ClassVar, Dict, List, Type

import numpy as np
import pandas as pd
import vectorbt as vbt
from alpaca.data.timeframe import TimeFrameUnit
from pydantic import BaseModel, Field, model_validator
from scipy.spatial.distance import mahalanobis


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
    source: str = Field(..., description="Source file of the feature data")
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
    def factory(cls, data: dict) -> "Feature":
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
        return subclass(**data)


class Candle(Feature):
    """
    Represents a candle feature with OHLCV data.
    """

    def get_feature_names(self) -> List[str]:
        return ["open", "high", "low", "close", "volume"]

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        return self.clean_columns(self.get_feature_names(), data)

    TYPE: ClassVar[FeatureType] = FeatureType.CANDLE
    open: float = Field(default=0.0, description="Open price of the candle")
    high: float = Field(default=0.0, description="High price of the candle")
    low: float = Field(default=0.0, description="Low price of the candle")
    close: float = Field(default=0.0, description="Close price of the candle")
    volume: float = Field(default=0.0, description="Volume of the candle")
    period: str = Field(
        TimeFrameUnit.Day, description="Time period of the candle (e.g., '1d', '1h')"
    )


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

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        atr = vbt.ATR.run(
            high=df[self.field_high],
            low=df[self.field_low],
            close=df[self.field_close],
            window=self.period,
        ).atr
        df[self.name] = atr
        return self.clean_columns(self.get_feature_names(), df)


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
        return [f"{self.name}", f"{self.name}_signal", f"{self.name}_hist"]

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
        return self.clean_columns(self.get_feature_names(), df)


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
        return [f"{self.name}_upper", f"{self.name}_lower", f"{self.name}_mid"]

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
        return self.clean_columns(self.get_feature_names(), df)


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
        )
        return self.clean_columns(self.get_feature_names(), df)


class OBV(Feature):
    """
    On-Balance Volume (OBV) feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.OBV
    field_close: str = Field("close", description="Field for close prices")
    field_volume: str = Field("volume", description="Field for volume")

    def to_df(self, df: pd.DataFrame, data: Any) -> pd.DataFrame:
        df[self.name] = vbt.OBV.run(
            close=df[self.field_close],
            volume=df[self.field_volume],
        )
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
    Computes a 'turbulence index' indicating how unusual current returns are
    compared to the recent historical return distribution.
    """

    TYPE: ClassVar[FeatureType] = FeatureType("turbulence")
    lookback: int = Field(20, description="Lookback window for calculating turbulence")
    field: str = Field("close", description="Field to use for return computation")

    def to_df(self, df: pd.DataFrame, data: Any = None) -> pd.DataFrame:
        returns = df[self.field].pct_change().dropna()
        turbulence_values = [np.nan] * self.lookback  # Start with NaNs

        for i in range(self.lookback, len(returns)):
            window = returns[i - self.lookback : i]
            current = returns.iloc[i]

            # Avoid degenerate case
            if window.std() == 0 or len(window) < 2:
                turbulence_values.append(0)
                continue

            mu = window.mean()
            cov = window.std() ** 2

            # Mahalanobis distance with 1D standard deviation as denominator
            dist = (current - mu) / np.sqrt(cov)
            turbulence_values.append(dist**2)  # Squared distance

        # Add to df
        df[self.name] = [np.nan] + turbulence_values  # Align with df index
        return self.clean_columns(self.get_feature_names(), df)
