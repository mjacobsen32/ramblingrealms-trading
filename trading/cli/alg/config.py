import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Union

from alpaca.data.timeframe import TimeFrameUnit
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo

from trading.src.features.generic_features import Feature
from trading.src.user_cache.user_cache import UserCache as user_cache


class ProjectPath(BaseModel):
    """
    Path-like object for using variable based absolute paths

    """

    PROJECT_ROOT: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent
    OUT_DIR: ClassVar[Path | None] = None
    BACKTEST_DIR: ClassVar[Path | None] = None
    VERSION: ClassVar[str] = str()
    ACTIVE_UUID: ClassVar[uuid.UUID | None] = None
    path: str = Field(str(), description="Path to the project root")

    @classmethod
    def use_cache(cls):
        cache = user_cache.load()
        ProjectPath.OUT_DIR = cache.out_dir
        ProjectPath.BACKTEST_DIR = cache.backtest_dir

    @classmethod
    def cache(cls):
        # persist output directories to user cache
        cache = user_cache.load()
        cache.out_dir = cls.OUT_DIR if cls.OUT_DIR else Path()
        cache.backtest_dir = cls.BACKTEST_DIR if cls.BACKTEST_DIR else Path()
        cache.save()

    @model_validator(mode="before")
    @classmethod
    def validate_path(cls, data) -> dict[str, str]:
        # Accept both dict and str
        if isinstance(data, dict):
            value = data.get("path", "")
        else:
            value = str(data)
        if "{PROJECT_ROOT}" in value:
            value = value.replace("{PROJECT_ROOT}", str(cls.PROJECT_ROOT))
        elif "{OUT_DIR}" in value:
            value = value.replace("{OUT_DIR}", str(cls.OUT_DIR))
        elif "{BACKTEST_DIR}" in value:
            value = value.replace("{BACKTEST_DIR}", str(cls.BACKTEST_DIR))
        if "{VERSION}" in value:
            value = value.replace("{VERSION}", str(cls.VERSION))
        if "{TIMESTAMP}" in value:
            value = value.replace(
                "{TIMESTAMP}", str(datetime.now().strftime("%Y%m%d_%H%M%S"))
            )
        if "{UUID}" in value:
            if cls.ACTIVE_UUID is None:
                raise ValueError(
                    "ACTIVE_UUID is not set, cannot replace {UUID} in path."
                )
            value = value.replace("{UUID}", str(cls.ACTIVE_UUID))
        return {"path": value}

    def __str__(self) -> str:
        return self.path

    def as_path(self) -> Path:
        p = Path(self.path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        return p


class DataSourceType(str, Enum):
    """
    Enum for data source types.
    """

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
        default_factory=list, description="List of feature names"
    )
    fill_strategy: str = Field(
        "mean",
        description="Strategy for handling missing values (e.g., 'mean', 'median', 'drop')",
    )

    def __repr__(self):
        return f"FeatureConfig(features={self.features}, fill_strategy={self.fill_strategy})"


class DataRequests(BaseModel):
    """
    Individual Data Request
    """

    dataset_name: str = Field("Generic", description="Name of the dataset")
    source: DataSourceType = Field(..., description="DataSourceType Enum")
    endpoint: str = Field(..., description="Endpoint of API")
    kwargs: Dict[str, Any] = Field(..., description="Kwargs to pass in to the endpoint")


class DataConfig(BaseModel):
    """
    Configuration for data used in the algorithm.
    """

    start_date: str = Field(
        "2020-01-01", description="Start date for the data collection"
    )
    end_date: str = Field("2023-01-01", description="End date for the data collection")
    time_step_unit: TimeFrameUnit = Field(
        TimeFrameUnit.Day, description="Time step of the data"
    )
    time_step_period: int = Field(
        1, description="Period of the time step (e.g., 1 for daily, 5 for 5-minute)"
    )
    cache_path: ProjectPath = Field(
        default_factory=lambda: ProjectPath.model_construct(),
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
    def validate_time_step_unit(cls, value: TimeFrameUnit | str) -> TimeFrameUnit:
        """
        Validate the time_step_unit field to ensure it is a valid TimeFrameUnit.
        """
        try:
            return TimeFrameUnit(value)
        except ValueError:
            raise ValueError(
                f"Invalid time step: {value}. Must be a valid TimeFrameUnit: {', '.join([unit.value for unit in TimeFrameUnit])}"
            )


class RewardConfig(BaseModel):
    """
    Configuration for the reward function used in the trading environment.
    """

    type: str = Field("basic_profit_max", description="Type of reward function to use")
    reward_scaling: float = Field(
        1e4, description="Scaling factor for the reward function"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the reward function",
    )


class TradeMode(str, Enum):
    """
    Enum for trade modes.
    """

    DISCRETE = "discrete"
    CONTINUOUS = "cont"


class PortfolioConfig(BaseModel):
    initial_cash: int = Field(100_000, description="Starting funds in state space")
    maintain_history: bool = Field(
        True, description="Whether to maintain a history of past actions and states"
    )
    buy_cost_pct: Union[float, List[float]] = Field(
        0.00, description="Corresponding cost for all assets or array per symbol"
    )
    sell_cost_pct: Union[float, List[float]] = Field(
        default=0.00,
        description="Corresponding cost for all assets or array per symbol",
    )

    # ! THESE ARE PSUEDO-MODEL-HYPERPARAMETERS, THEY MUST BE MAINTAINED ACCROSS TRAINING, TESTING, AND PRODUCTION ENVIRONMENTS
    max_positions: int | None = Field(
        None,
        description="Maximum number of open positions per asset at any time. If None, no limit is applied. Does not apply to Continuous action space",
    )
    trade_mode: TradeMode = Field(
        TradeMode.CONTINUOUS,
        description="Mode for trading: DISCRETE (fixed actions) or CONTINUOUS (scaled based on actions)",
    )
    trade_limit_percent: float = Field(
        0.1,
        description="Maximum percentage of cash to be used in a single trade relative to the current asset value",
    )
    hmax: float = Field(
        10_000.0,
        description="Maximum cash to be traded in each trade per asset, if using discrete actions each trade is the max.",
    )
    action_threshold: float = Field(
        0.1,
        description="Minimum action value to trigger a trade, used to avoid noise in continuous actions",
    )
    max_exposure: float = Field(
        1.0, description="Maximum exposure across the entire portfolio"
    )
    # ! THESE ARE PSUEDO-MODEL-HYPERPARAMETERS, THEY MUST BE MAINTAINED ACCROSS TRAINING, TESTING, AND PRODUCTION ENVIRONMENTS


class StockEnv(BaseModel):
    turbulence_threshold: float | None = Field(
        None,
        description="Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated",
    )
    reward_config: RewardConfig = Field(
        default_factory=lambda: RewardConfig.model_construct(),
        description="Reward function configuration",
    )
    portfolio_config: PortfolioConfig = Field(
        default_factory=lambda: PortfolioConfig.model_construct(),
        description="Portfolio configuration",
    )
    lookback_window: int = Field(
        10, description="Number of past timesteps to consider for state representation"
    )


class AgentConfig(BaseModel):
    """
    Configuration for the agent used in the algorithm.
    """

    algo: str = Field(
        "ppo", description="Algorithm to use (e.g., 'ppo', 'a2c', 'dqn', etc.)"
    )
    save_path: ProjectPath = Field(
        default_factory=lambda: ProjectPath.model_construct(),
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


class AnalysisConfig(BaseModel):
    """
    Configuration for the analysis of backtest results.
    """

    render_plots: bool = Field(
        True, description="Plot the backtesting results in browser using .show()"
    )
    save_plots: bool = Field(True, description="Whether to save the analysis plots")
    to_csv: bool = Field(True, description="Save the analysis results to CSV files")
    tickers: List[str] | None = Field(
        None,
        description="List of tickers to generate plots for. If None, generate for all.",
    )


class BackTestConfig(BaseModel):
    """
    Configuration for backtesting the algorithm.
    """

    save_results: bool = Field(
        False, description="Whether to save the backtesting results"
    )
    backtest_dir: ProjectPath = Field(
        default_factory=lambda: ProjectPath.model_construct(),
        description="Directory for storing backtest results",
    )
    analysis_config: AnalysisConfig = Field(
        default_factory=lambda: AnalysisConfig.model_construct(),
    )

    @field_validator(
        "backtest_dir",
        mode="after",
    )
    @classmethod
    def validate_backtest_dir(cls, value: ProjectPath) -> BaseModel:
        logging.info("Validating backtest directory: %s", value)
        p = ProjectPath.model_validate(value)
        ProjectPath.BACKTEST_DIR = p.as_path()
        ProjectPath.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        logging.info("Backtest results will be saved to: %s", ProjectPath.BACKTEST_DIR)
        return ProjectPath.model_validate(p)

    results_path: ProjectPath = Field(
        default_factory=lambda: ProjectPath.model_construct(),
        description="Path to save the backtesting results",
    )


class RRConfig(BaseModel):
    """
    Configuration for the algorithm.
    """

    name: str = Field(..., description="Name of the algorithm")
    description: str = Field("", description="Description of the algorithm")
    version: str = Field(str(), description="Version of the algorithm")
    out_dir: ProjectPath = Field(default_factory=lambda: ProjectPath.model_construct())
    agent_config: AgentConfig = Field(
        default_factory=lambda: AgentConfig.model_construct(),
        description="Agent configuration",
    )
    data_config: DataConfig = Field(
        default_factory=lambda: DataConfig.model_construct(),
        description="Data configuration",
    )
    feature_config: FeatureConfig = Field(
        default_factory=lambda: FeatureConfig.model_construct(),
        description="Feature configuration",
    )
    stock_env: StockEnv = Field(
        default_factory=lambda: StockEnv.model_construct(),
        description="Stock Trading Environment Config",
    )
    backtest_config: BackTestConfig = Field(
        default_factory=lambda: BackTestConfig.model_construct(),
        description="Backtesting configuration",
    )

    @field_validator(
        "version",
        mode="before",
    )
    @classmethod
    def validate_version(cls, value: str) -> str:
        ProjectPath.VERSION = value
        return value

    @field_validator(
        "out_dir",
        mode="before",
    )
    @classmethod
    def validate_out_dir(cls, value: ProjectPath) -> BaseModel:
        p = ProjectPath.model_validate(value)
        if ProjectPath.OUT_DIR is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ProjectPath.OUT_DIR = p.as_path() / Path(timestamp)
            ProjectPath.OUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info("Saving/loading all output to/from: %s", ProjectPath.OUT_DIR)
        return ProjectPath.model_validate(p)

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
        except ValidationError:
            if isinstance(value, BaseModel):
                try:
                    return string_map[str(info.field_name)].model_validate(value)
                except ValidationError as e_two:
                    logging.error("Validation error two: %s", e_two)
        return value
