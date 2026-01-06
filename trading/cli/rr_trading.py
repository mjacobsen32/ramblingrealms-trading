import datetime
import logging
import uuid
from pathlib import Path
from typing import Annotated

import typer
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient as AlpacaTradingClient
from pydantic import SecretStr
from rich.prompt import Prompt

from trading.cli.alg import alg
from trading.cli.data import data
from trading.cli.trading.trade_config import ProjectPath, RRTradeConfig
from trading.cli.utils import init_logger
from trading.src.trade.trade_api import Trade
from trading.src.user_cache.user_cache import UserCache as User
from trading.src.utility.utils import read_key

app = typer.Typer(name="rr_trading", help="rr_trading CLI commands")
app.add_typer(data.app, name="data", help="Data CLI commands")
app.add_typer(alg.app, name="alg", help="Algorithmic commands")


class AppState:
    def __init__(self):
        self.file_log_level = "INFO"
        self.console_log_level = "INFO"


FORMAT = "%(message)s"


@app.callback()
def rr_trading(
    ctx: typer.Context,
    log_level_console: str = typer.Option(
        "INFO",
        "--log-level-console",
        help="Logging level for console (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET)",
    ),
    log_level_file: str = typer.Option(
        "NOTSET",
        "--log-level-file",
        help="Logging level for file log (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET)",
    ),
):
    """Initialize logging before any command runs."""
    # Allow options to be parsed from anywhere in the command line
    ctx.resilient_parsing = True
    ctx.ensure_object(AppState)
    ctx.obj.file_log_level = log_level_file.upper()
    ctx.obj.console_log_level = log_level_console.upper()
    init_logger(ctx.obj.console_log_level, ctx.obj.file_log_level)


@app.command(help="Print the current user configuration")
def print_config():
    """
    Print the current user configuration.
    """
    config = User.load()
    logging.info(config)


@app.command(help="Run the setup wizard")
def setup():
    """
    Interactive setup wizard for the user configuration.
    """
    user = User.load()
    user.delete()

    polygon = Prompt.ask("Do you want to set up Polygon?", default="n")
    if polygon.lower() == "y":
        # Polygon setup
        user.polygon_access_token_path = Path(
            Prompt.ask("Enter your Polygon access token path")
        )
    alpaca = Prompt.ask("Do you want to set up Alpaca?", default="n")
    if alpaca.lower() == "y":
        # Alpaca setup
        user.alpaca_api_key = SecretStr(
            read_key(Prompt.ask("Enter your Alpaca API key path"))
        )
        user.alpaca_api_secret = SecretStr(
            read_key(Prompt.ask("Enter your Alpaca API secret path"))
        )
    alpaca_live = Prompt.ask("Do you want to set up Alpaca Live Trading?", default="n")
    if alpaca_live.lower() == "y":
        # Alpaca Live Trading setup
        user.alpaca_api_key_live = SecretStr(
            read_key(Prompt.ask("Enter your Alpaca API key path for Live Trading"))
        )
        user.alpaca_api_secret_live = SecretStr(
            read_key(Prompt.ask("Enter your Alpaca API secret path for Live Trading"))
        )
    remote_portfolio = Prompt.ask(
        "Do you want to set up remote portfolio management with an S3 Client? (requires write access)",
        default="n",
    )
    if remote_portfolio.lower() == "y":
        # Remote portfolio setup
        user.r2_access_key_id = SecretStr(
            read_key(Prompt.ask("Enter your S3 Client Access Key ID (PUBLIC) key path"))
        )
        user.r2_secret_access_key = SecretStr(
            read_key(
                Prompt.ask(
                    "Enter your S3 Client Secret Access Key ID (PRIVATE) key path"
                )
            )
        )
        user.r2_endpoint_url = Prompt.ask("Enter your S3 Client Endpoint URL")


@app.command(help="Run model on paper trading Alpaca Account")
def paper_trade(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to the RRTrade configuration file."),
    ],
    predict_time_str: Annotated[
        str,
        typer.Option(
            "--timestamp",
            "-t",
            help="Timestamp for which to run the model (YYYY-MM-DD).",
        ),
    ] = "",
    predict_time_end_str: Annotated[
        str,
        typer.Option(
            "--timestamp_end",
            "-e",
            help="Timestamp for which to end if wanting a forward testing range: (YYYY-MM-DD).",
        ),
    ] = "",
    account_uuid: Annotated[
        str,
        typer.Option(
            "--uuid",
            "-u",
            help="UUID for the trading account.",
        ),
    ] = "",
):
    """
    Run the model on the Alpaca paper trading account.
    Explicit command for paper and live trading for total seperation
    """
    logging.info("Running model on Alpaca paper trading account...")
    if account_uuid != "":
        ProjectPath.ACTIVE_UUID = uuid.UUID(account_uuid)
        logging.info(f"Using ACCOUNT UUID: {ProjectPath.ACTIVE_UUID}")
    with Path.open(Path(config)) as f:
        rr_trade_config = RRTradeConfig.model_validate_json(f.read())
        logging.info(f"Loaded configuration from {config}")

    user_cache = User().load()
    alpaca_api_key = user_cache.alpaca_api_key
    alpaca_api_secret = user_cache.alpaca_api_secret

    market_data_client = StockHistoricalDataClient(
        alpaca_api_key.get_secret_value(), alpaca_api_secret.get_secret_value()
    )

    alpaca_account_client: AlpacaTradingClient = AlpacaTradingClient(
        alpaca_api_key.get_secret_value(),
        alpaca_api_secret.get_secret_value(),
        paper=True,
    )
    if predict_time_str == "":
        predict_time = datetime.datetime.now(tz=datetime.timezone.utc)
    else:
        predict_time = datetime.datetime.fromisoformat(predict_time_str).replace(
            tzinfo=datetime.timezone.utc
        )

    if predict_time_end_str == "":
        predict_time_end = predict_time
    else:
        predict_time_end = datetime.datetime.fromisoformat(
            predict_time_end_str
        ).replace(tzinfo=datetime.timezone.utc)

    trade_client = Trade(
        config=rr_trade_config,
        market_data_client=market_data_client,
        alpaca_account_client=alpaca_account_client,
        live=False,
        predict_time=predict_time,
        end_predict_time=predict_time_end,
    )

    try:
        trade_client.run_model(
            predict_time=predict_time, end_predict_time=predict_time_end
        )
    except Trade.LiveTradeError as e:
        if e.error_type == Trade.LiveTradeError.OUT_OF_RANGE:
            logging.warning("Trade skipped: %s", e.message)
        else:
            raise
    except Exception as e:
        logging.error("Unexpected error during trade execution: %s", str(e))
        raise


@app.command(help="Run model on live trading Alpaca Account")
def live_trade(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to the RRTrade configuration file."),
    ],
    confirmation: Annotated[
        bool, typer.Option("--confirm", "-y", help="Confirm live trading execution.")
    ] = False,
):
    """
    Run the model on the Alpaca live trading account.
    Explicit command for paper and live trading for total seperation
    """
    logging.warning("Running model on Alpaca live trading account...")
    confirmation_str = Prompt.ask(
        "[red]Are you sure you want to proceed with live trading?[/red]", default="n"
    )
    if not confirmation and confirmation_str.lower() != "y":
        logging.info("Live trading execution cancelled.")
        return
    logging.info("Running model on Alpaca live trading account...")

    with Path.open(Path(config)) as f:
        rr_trade_config = RRTradeConfig.model_validate_json(f.read())
        logging.info(f"Loaded configuration from {config}")

    user_cache = User().load()

    alpaca_api_key = user_cache.alpaca_api_key_live
    alpaca_api_secret = user_cache.alpaca_api_secret_live

    market_data_client = StockHistoricalDataClient(
        alpaca_api_key.get_secret_value(), alpaca_api_secret.get_secret_value()
    )

    alpaca_account_client: AlpacaTradingClient = AlpacaTradingClient(
        alpaca_api_key.get_secret_value(),
        alpaca_api_secret.get_secret_value(),
        paper=True,
    )

    predict_time = datetime.datetime.now()
    predict_time_end = predict_time

    trade_client = Trade(
        config=rr_trade_config,
        market_data_client=market_data_client,
        alpaca_account_client=alpaca_account_client,
        live=True,
        predict_time=predict_time,
        end_predict_time=predict_time_end,
    )

    try:
        trade_client.run_model(
            predict_time=predict_time, end_predict_time=predict_time_end
        )
    except Trade.LiveTradeError as e:
        if e.error_type == Trade.LiveTradeError.OUT_OF_RANGE:
            logging.warning("Trade skipped: %s", e.message)
        else:
            raise
    except Exception as e:
        logging.error("Unexpected error during trade execution: %s", str(e))
        raise


if __name__ == "__main__":
    app()
