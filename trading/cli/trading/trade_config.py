from enum import Enum

from pydantic import BaseModel, Field

from trading.cli.alg.config import PortfolioConfig, ProjectPath


class BrokerType(str, Enum):
    """
    Enum for broker types.
    """

    # ALPACA will connect to the ALPACA broker to executre trades and manage positions
    ALPACA = "ALPACA"
    # LOCAL will use a local simulated broken for paper trading
    LOCAL = "LOCAL"
    # REMOTE will use a remote simulated broker for paper trading, this mode is used to push
    # positions and trade activity to the S3 bucket consumed by the front-end dashboard
    REMOTE = "REMOTE"


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
