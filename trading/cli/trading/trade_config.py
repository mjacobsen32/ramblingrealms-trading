from enum import Enum

from pydantic import BaseModel, Field

from trading.cli.alg.config import PortfolioConfig, ProjectPath


class BrokerType(str, Enum):
    """
    Enum for broker types.
    """

    ALPACA = "ALPACA"
    LOCAL = "LOCAL"


class RRTradeConfig(BaseModel):
    """
    Trading Configuration
    TODO:
        - notifications
        - require trade approval
    """

    model_path: ProjectPath = Field(
        default_factory=ProjectPath, description="Path to the trained model."
    )
    broker: BrokerType = Field(
        BrokerType.LOCAL, description="Broker to use for trading."
    )
    broker_kwargs: dict = Field(
        default_factory=dict, description="Additional broker-specific arguments."
    )
    out_dir: ProjectPath = Field(
        default_factory=ProjectPath, description="Path to the output directory."
    )
    portfolio_config: PortfolioConfig | None = Field(
        None,
        description="Portfolio configuration. If None, the configuration saved with the model will be used",
    )
