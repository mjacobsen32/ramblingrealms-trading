import os
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Self, Union

from alpaca.data.timeframe import TimeFrameUnit
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo

from trading.src.features.generic_features import Feature


class ProjectPath(BaseModel):
    PROJECT_ROOT: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent
    path: str = Field(str(), description="Path to the project root")

    @model_validator(mode="before")
    @classmethod
    def validate_path(cls, data):
        # Accept both dict and str
        if isinstance(data, dict):
            value = data.get("path", "")
        else:
            value = str(data)
        if "{PROJECT_ROOT}" in value:
            value = value.replace("{PROJECT_ROOT}", str(cls.PROJECT_ROOT))
        return {"path": value}

    def __str__(self) -> str:
        return self.path

    def as_path(self) -> Path:
        return Path(self.path)


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
            (
                Feature.factory(f, fill_strategy=data.get("fill_strategy"))
                if isinstance(f, dict)
                else f
            )
            for f in features_data
        ]
        return data

    features: List[Feature] = Field(
        default_factory=List[Feature], description="List of feature names"
    )
    fill_strategy: str = Field(
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
    cache_path: ProjectPath = Field(
        default_factory=ProjectPath,
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
    trade_limit_percent: float = Field(
        0.1,
        description="Maximum percentage of cash to be used in a single trade relative to the current asset value",
    )


class AgentConfig(BaseModel):
    """
    Configuration for the agent used in the algorithm.
    """

    algo: str = Field(
        "ppo", description="Algorithm to use (e.g., 'ppo', 'a2c', 'dqn', etc.)"
    )
    save_path: ProjectPath = Field(
        default_factory=ProjectPath,
        description="Path to save the trained agent model",
    )
    deterministic: bool = Field(
        True, description="Whether to use deterministic actions during inference"
    )
    total_timesteps: int = Field(
        1_000_000, description="Total number of timesteps for training the agent"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters for the agent"
    )
    log_dir: ProjectPath | None = Field(
        None,
        description="Directory to save logs and model checkpoints",
    )


class BackTestConfig(BaseModel):
    """
    Configuration for backtesting the algorithm.
    """

    save_results: bool = Field(
        False, description="Whether to save the backtesting results"
    )
    results_path: ProjectPath = Field(
        default_factory=ProjectPath,
        description="Path to save the backtesting results",
    )


class RRConfig(BaseModel):
    """
    Configuration for the algorithm.
    """

    name: str = Field(..., description="Name of the algorithm")
    description: str = Field("", description="Description of the algorithm")
    version: str = Field("1.0.0", description="Version of the algorithm")
    agent_config: AgentConfig | ProjectPath = Field(
        default_factory=AgentConfig, description="Agent configuration"
    )
    data_config: DataConfig | ProjectPath = Field(
        default_factory=DataConfig, description="Data configuration"
    )
    feature_config: FeatureConfig | ProjectPath = Field(
        default_factory=FeatureConfig, description="Feature configuration"
    )
    stock_env: StockEnv | ProjectPath = Field(
        default_factory=StockEnv, description="Stock Trading Environment Config"
    )
    log_dir: ProjectPath = Field(default_factory=ProjectPath)
    backtest_config: BackTestConfig | ProjectPath = Field(
        default_factory=BackTestConfig, description="Backtesting configuration"
    )

    @field_validator(
        "agent_config",
        "data_config",
        "feature_config",
        "stock_env",
        mode="before",
    )
    @classmethod
    def validate_config(
        cls, value: Union[ProjectPath, BaseModel], info: ValidationInfo
    ) -> BaseModel:
        string_map: dict[str, type[BaseModel]] = {
            "agent_config": AgentConfig,
            "data_config": DataConfig,
            "feature_config": FeatureConfig,
            "stock_env": StockEnv,
        }
        try:
            value = ProjectPath.model_validate(value)
            if isinstance(value, ProjectPath):
                return string_map[str(info.field_name)].model_validate_json(
                    value.as_path().read_text()
                )
        except ValidationError as e_one:
            if isinstance(value, BaseModel):
                try:
                    return string_map[str(info.field_name)].model_validate(value)
                except ValidationError as e_two:
                    print("Validation error two:", e_two)
        return value
