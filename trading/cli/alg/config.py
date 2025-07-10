from enum import Enum
from typing import Any, Dict, List, Union

from alpaca.data.timeframe import TimeFrameUnit
from pydantic import BaseModel, Field, field_validator, model_validator

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
        data["features"] = [
            Feature.factory(f) if isinstance(f, dict) else f for f in features_data
        ]
        return data

    features: List[Feature] = Field(
        default_factory=List[Feature], description="List of feature names"
    )
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
    kwargs: Dict = Field(..., description="Kwargs to pass in to the endpoint")


class DataConfig(BaseModel):
    """
    Configuration for data used in the algorithm.
    """

    start_date: str = Field(
        "2020-01-01", description="Start date for the data collection"
    )
    end_date: str = Field("2023-01-01", description="End date for the data collection")
    time_step_unit: str = Field(TimeFrameUnit.Day, description="Time step of the data")
    time_step_period: int = Field(
        1, description="Period of the time step (e.g., 1 for daily, 5 for 5-minute)"
    )
    cache_path: str = Field(
        "cache/",
        description="Path to cache the downloaded data",
    )
    cache_enabled: bool = Field(
        True, description="Whether to cache the downloaded data"
    )
    requests: List[DataRequests] = Field(..., description="List of data requests")
    validation_split: float = Field(
        0.2, description="Fraction of data to use for validation"
    )

    @field_validator("time_step_unit")
    @classmethod
    def validate_time_step_unit(cls, value: str) -> TimeFrameUnit:
        """
        Validate the time_step_unit field to ensure it is a valid TimeFrameUnit.
        """
        try:
            return TimeFrameUnit(value)
        except ValueError:
            raise ValueError(
                f"Invalid time step: {value}. Must be a valid TimeFrameUnit: {', '.join([unit.value for unit in TimeFrameUnit])}"
            )


class TrainConfig(BaseModel):
    """
    Configuration for training the algorithm.
    """

    epochs: int = Field(100, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size for training")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer")
    early_stopping_patience: int = Field(10, description="Patience for early stopping")


class StockEnv(BaseModel):
    initial_cash: int = Field(100_000, description="Starting funds in state space")
    hmax: int = Field(
        10_000, description="Maximum cash to be traded in each trade per asset"
    )
    buy_cost_pct: Union[float, List[float]] = Field(
        0.00, description="Corresponding cost for all assets or array per symbol"
    )
    sell_cost_pct: Union[float, List[float]] = Field(
        0.00, description="Corresponding cost for all assets or array per symbol"
    )
    turbulence_threshold: Union[float, None] = Field(
        None,
        description="Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated",
    )
    reward_scaling: float = Field(1e-4)


class AgentConfig(BaseModel):
    """
    Configuration for the agent used in the algorithm.
    """

    algo: str = Field(
        "ppo", description="Algorithm to use (e.g., 'ppo', 'a2c', 'dqn', etc.)"
    )
    save_path: str = Field(
        "models/",
        description="Path to save the trained agent model",
    )
    deterministic: bool = Field(
        True, description="Whether to use deterministic actions during inference"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters for the agent"
    )


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
    agent_config: AgentConfig = Field(
        default_factory=AgentConfig, description="Agent configuration"
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
    stock_env: StockEnv = Field(
        default_factory=StockEnv, description="Stock Trading Environment Config"
    )
    output_dir: str = Field("./logs")
