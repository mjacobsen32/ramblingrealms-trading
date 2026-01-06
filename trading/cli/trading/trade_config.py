import logging
import uuid
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

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


class AssetExchange(str, Enum):
    """
    Represents the current exchanges Alpaca supports.
    """

    AMEX = "AMEX"
    ARCA = "ARCA"
    BATS = "BATS"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    NYSEARCA = "NYSEARCA"
    FTXU = "FTXU"
    CBSE = "CBSE"
    GNSS = "GNSS"
    ERSX = "ERSX"
    OTC = "OTC"
    CRYPTO = "CRYPTO"
    EMPTY = ""


class RRTradeConfig(BaseModel):
    """
    Trading Configuration
    TODO:
        - notifications
        - require trade approval
    """

    id: UUID | None = Field(
        default_factory=uuid.uuid4,
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
    closed_positions_path: ProjectPath | None = Field(
        default=None,
        description="Path to closed positions JSON for local/remote brokers.",
    )
    account_value_series_path: ProjectPath | None = Field(
        default=None,
        description="Path to account value series JSON for local/remote brokers.",
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
    asset_exchanges: list[AssetExchange] = Field(
        default=[AssetExchange.NYSE, AssetExchange.NASDAQ],
        description="List of asset exchanges to filter tradable assets. If None, all exchanges are considered.",
    )

    @model_validator(mode="before")
    def ensure_id(cls, values):
        id_value = values.get("id")
        if id_value is None and ProjectPath.ACTIVE_UUID is None:
            new_id = uuid.uuid4()
            ProjectPath.ACTIVE_UUID = new_id
            logging.warning(
                "No ACCOUNT UUID provided, generating a new one: %s\n"
                "If you want to connect to a specific remote account please provide a uuid via command line or in the config",
                new_id,
            )
            values["id"] = new_id
        if id_value is None:
            values["id"] = ProjectPath.ACTIVE_UUID
        return values
