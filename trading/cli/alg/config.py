from pydantic import BaseModel, Field, field_validator
from typing import Union, List

from trading.src.features.generic_features import Feature


class FeatureConfig(BaseModel):
    """
    Configuration for features used in the algorithm.
    """

    features: List[Feature] = Field(..., description="List of feature names")
    normalization: bool = Field(True, description="Whether to normalize features")
    missing_value_strategy: str = Field(
        "mean",
        description="Strategy for handling missing values (e.g., 'mean', 'median', 'drop')",
    )


class DataConfig(BaseModel):
    """
    Configuration for data used in the algorithm.
    """

    dataset_name: str = Field("Generic", description="Name of the dataset")
    cache_path: str = Field(
        "cache/",
        description="Path to cache the downloaded data",
    )
    cache_enabled: bool = Field(
        True, description="Whether to cache the downloaded data"
    )
    time_step: str = Field("1d", description="Time step of the data (e.g., '1d', '1h')")
    start_date: str = Field(
        "2020-01-01", description="Start date for the data collection"
    )
    end_date: str = Field("2023-01-01", description="End date for the data collection")
    tickers: Union[List[str], str] = Field(
        ["AAPL", "GOOGL", "MSFT"],
        description='List of stock tickers to include in the dataset, or "ALL" for all available tickers',
    )


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
