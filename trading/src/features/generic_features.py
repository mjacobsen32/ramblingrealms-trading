from enum import Enum
from typing import ClassVar, Dict, Literal, Type

from alpaca.data.timeframe import TimeFrameUnit
from pydantic import BaseModel, Field


class FeatureType(str, Enum):
    """
    Enum for feature types.
    """

    CANDLE = "candle"
    MOVING_AVERAGE = "moving_average"
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

    TYPE: ClassVar[FeatureType] = FeatureType.CANDLE
    open: float = Field(..., description="Open price of the candle")
    high: float = Field(..., description="High price of the candle")
    low: float = Field(..., description="Low price of the candle")
    close: float = Field(..., description="Close price of the candle")
    volume: float = Field(..., description="Volume of the candle")
    period: TimeFrameUnit = Field(
        TimeFrameUnit.Day, description="Time period of the candle (e.g., '1d', '1h')"
    )


class MovingAverage(Feature):
    """
    Represents a moving average feature by Days.
    """

    TYPE: ClassVar[FeatureType] = FeatureType.MOVING_AVERAGE
    period: int = Field(
        ..., description="The period for the moving average (e.g., 20 for a 20-day MA)"
    )
