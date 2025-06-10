from enum import Enum
from typing import Any, Dict, List

from alpaca.data.timeframe import TimeFrameUnit
from pydantic import BaseModel, Field, model_validator

from trading.src.features.generic_features import Feature


class DataSourceType(str, Enum):
    ALPACA = "alpaca"


class FeatureConfig(BaseModel):
    """
    Configuration for features used in the algorithm.
    """

    @model_validator(mode="before")
    @classmethod
    def parse_features(cls, data: Any):
        features_data = data.get("features", [])
        data["features"] = [Feature.factory(f) for f in features_data]
        return data

    features: List[Feature] = Field(
        default_factory=Feature, description="List of feature names"
    )
    normalization: bool = Field(True, description="Whether to normalize features")
    missing_value_strategy: str = Field(
        "mean",
        description="Strategy for handling missing values (e.g., 'mean', 'median', 'drop')",
    )


class DataRequests(BaseModel):
    """
    Individual Data Request
    """

    dataset_name: str = Field("Generic", description="Name of the dataset")
    source: DataSourceType = Field(..., description="DataSourceType Enum")
    endpoint: str = Field(..., description="Endpoint of API")
    cache_enabled: bool = Field(
        True, description="Whether to cache the downloaded data"
    )
    kwargs: Dict = Field(..., description="Kwargs to pass in to the endpoint")


class DataConfig(BaseModel):
    """
    Configuration for data used in the algorithm.
    """

    start_date: str = Field(
        "2020-01-01", description="Start date for the data collection"
    )
    end_date: str = Field("2023-01-01", description="End date for the data collection")
    time_step: TimeFrameUnit = Field(
        TimeFrameUnit.Day, description="Time step of the data"
    )
    cache_path: str = Field(
        "cache/",
        description="Path to cache the downloaded data",
    )
    requests: List[DataRequests] = Field(..., description="List of data requests")


class TrainConfig(BaseModel):
    """
    Configuration for training the algorithm.
    """

    epochs: int = Field(100, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size for training")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer")
    validation_split: float = Field(
        0.2, description="Fraction of data to use for validation"
    )
    early_stopping_patience: int = Field(10, description="Patience for early stopping")


class AlgConfig(BaseModel):
    """
    Configuration for the algorithm.
    """

    name: str = Field(..., description="Name of the algorithm")
    description: str = Field("", description="Description of the algorithm")
    version: str = Field("1.0.0", description="Version of the algorithm")
    train_config: TrainConfig = Field(
        default_factory=TrainConfig, description="Training configuration"
    )
    data_config: DataConfig = Field(
        default_factory=DataConfig, description="Data configuration"
    )
    feature_config: FeatureConfig = Field(
        default_factory=FeatureConfig, description="Feature configuration"
    )
    save_path: str = Field(
        "models/",
        description="Path to save the trained model",
    )
