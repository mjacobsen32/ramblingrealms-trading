from enum import Enum

from pydantic import BaseModel, Field

from trading.cli.alg.config import ProjectPath, TradeMode


class BrokerType(str, Enum):
    """
    Enum for broker types.
    """

    ALPACA = "alpaca"


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
        BrokerType.ALPACA, description="Broker to use for trading."
    )
    broker_kwargs: dict = Field(
        default_factory=dict, description="Additional broker-specific arguments."
    )
    out_dir: ProjectPath = Field(
        default_factory=ProjectPath, description="Path to the output directory."
    )
    trade_mode: TradeMode = Field(
        TradeMode.CONTINUOUS,
        description="Mode for trading: DISCRETE (fixed actions) or CONTINUOUS (scaled based on actions). Must match model",
    )
    max_exposure: float = Field(
        default=1.0, description="Maximum exposure for entire portfolio."
    )
    max_positions: int | None = Field(
        default=None,
        description="Maximum number of open positions per asset. Should match training configurations but does not need to.",
    )
