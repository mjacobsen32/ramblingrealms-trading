from enum import Enum
from typing import Any, ClassVar, Dict, Type

import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit
from pydantic import BaseModel, Field, model_validator


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


class Feature(BaseModel):
    """
    Represents a single feature with its name and value.
    """

    type: FeatureType
    name: str = Field(..., description="Name of the feature")
    enabled: bool = Field(True, description="Whether the feature is enabled or not")
    source: str = Field(..., description="Source file of the feature data")

    # Registry for subclasses
    _registry: ClassVar[Dict[FeatureType, Type["Feature"]]] = {}

    def __init__(self, **data):
        if type(self) is Feature:
            raise TypeError(
                "Feature is an abstract base class and cannot be instantiated directly."
            )
        super().__init__(**data)

    def to_df(self, df: pd.DataFrame, data: Any) -> None:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        type_val = getattr(cls, "TYPE", None)
        if type_val is not None:
            cls._registry[type_val] = cls

    @classmethod
    def factory(cls, data: dict) -> "Feature":
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

    def to_df(self, df: pd.DataFrame, data: Any) -> None:
        df["open"] = data["open"]
        df["high"] = data["high"]
        df["low"] = data["low"]
        df["close"] = data["close"]
        df["volume"] = data["volume"]

    TYPE: ClassVar[FeatureType] = FeatureType.CANDLE
    open: float = Field(default=0.0, description="Open price of the candle")
    high: float = Field(default=0.0, description="High price of the candle")
    low: float = Field(default=0.0, description="Low price of the candle")
    close: float = Field(default=0.0, description="Close price of the candle")
    volume: float = Field(default=0.0, description="Volume of the candle")
    period: TimeFrameUnit = Field(
        TimeFrameUnit.Day, description="Time period of the candle (e.g., '1d', '1h')"
    )


class OperationType(str, Enum):
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


class MovingWindow(Feature):
    """
    Represents a moving average feature by Days.
    """

    def to_df(self, df: pd.DataFrame, data: Any) -> None:
        df[self.name] = self.operation(df, self.field, self.period)

    TYPE: ClassVar[FeatureType] = FeatureType.MOVING_WINDOW
    period: int = Field(
        ...,
        description="The period for the moving operation (e.g., 20 for a 20-day Moving (average))",
    )
    field: str = Field("close", description="Field to roll")
    operation: OperationType = Field(
        OperationType.MEAN, description="Operation to perform on rolling values"
    )


class RSI(Feature):
    """
    Relative Strength Index (RSI) feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.RSI
    period: int = Field(14, description="Period for RSI calculation")
    field: str = Field("close", description="Field to calculate RSI on")

    def to_df(self, df: pd.DataFrame, data: Any) -> None:
        delta = pd.to_numeric(df[self.field].diff(), errors="coerce")
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        df[self.name] = 100 - (100 / (1 + rs))


class MACD(Feature):
    """
    Moving Average Convergence Divergence (MACD) feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.MACD
    fast_period: int = Field(12)
    slow_period: int = Field(26)
    signal_period: int = Field(9)
    field: str = Field("close")

    def to_df(self, df: pd.DataFrame, data: Any) -> None:
        ema_fast = df[self.field].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df[self.field].ewm(span=self.slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        df[f"{self.name}_macd"] = macd
        df[f"{self.name}_signal"] = signal
        df[f"{self.name}_hist"] = macd - signal


class BollingerBands(Feature):
    """
    Bollinger Bands feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.BOLLINGER_BANDS
    period: int = Field(20)
    field: str = Field("close")

    def to_df(self, df: pd.DataFrame, data: Any) -> None:
        sma = df[self.field].rolling(window=self.period).mean()
        std = df[self.field].rolling(window=self.period).std()
        df[f"{self.name}_upper"] = sma + (2 * std)
        df[f"{self.name}_lower"] = sma - (2 * std)
        df[f"{self.name}_mid"] = sma


class Stochastic(Feature):
    """
    Stochastic Oscillator feature.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.STOCHASTIC
    period: int = Field(14)
    field_high: str = Field("high")
    field_low: str = Field("low")
    field_close: str = Field("close")

    def to_df(self, df: pd.DataFrame, data: Any) -> None:
        low_min = df[self.field_low].rolling(window=self.period).min()
        high_max = df[self.field_high].rolling(window=self.period).max()
        df[self.name] = 100 * (
            (df[self.field_close] - low_min) / (high_max - low_min + 1e-10)
        )
