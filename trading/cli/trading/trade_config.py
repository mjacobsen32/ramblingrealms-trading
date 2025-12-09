from enum import Enum
from uuid import UUID

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

    id: UUID = Field(
        default=UUID("00000000-0000-0000-0000-000000000001"),
        description="User ID for the trading account.",
    )
    account_number: str = Field(
        default="000-000-000", description="Account number for the trading account."
    )
    model_path: ProjectPath = Field(
        default_factory=lambda: ProjectPath.model_construct(),
        description="Path to the trained model.",
    )
    broker: BrokerType = Field(
        default=BrokerType.LOCAL, description="Broker to use for trading."
    )
    broker_kwargs: dict = Field(
        default_factory=dict, description="Additional broker-specific arguments."
    )
    bucket_name: str | None = Field(
        default=None,
        description="Name of the remote S3 bucket for REMOTE broker.",
    )
    positions_path: ProjectPath | None = Field(
        default=None,
        description="Path to positions JSON for local/remote brokers.",
    )
    account_path: ProjectPath | None = Field(
        default=None,
        description="Path to account JSON for local/remote brokers.",
    )
    out_dir: ProjectPath = Field(
        default_factory=lambda: ProjectPath.model_construct(),
        description="Path to the output directory.",
    )
    portfolio_config: PortfolioConfig | None = Field(
        default=None,
        description="Portfolio configuration. If None, the configuration saved with the model will be used",
    )
    defer_trade_execution: bool = Field(
        default=False,
        description="If True, trade executions will be deferred until program termination, utilizing batched writes.",
    )
