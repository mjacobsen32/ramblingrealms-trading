import logging
from pathlib import Path

import typer
from pydantic import SecretStr
from rich import print
from rich.logging import RichHandler
from rich.prompt import Prompt

from trading.cli.alg import alg
from trading.cli.data import data
from trading.cli.etrade import etrade
from trading.src.user_cache.user_cache import UserCache as User
from trading.src.utility.utils import read_key

app = typer.Typer(name="rr_trading", help="rr_trading CLI commands")
app.add_typer(etrade.app, name="etrade", help="E-Trade API commands")
app.add_typer(data.app, name="data", help="Data CLI commands")
app.add_typer(alg.app, name="alg", help="Algorithmic commands")


@app.callback()
def main(
    ctx: typer.Context,
    log_level_console: str = typer.Option(
        "INFO",
        "--log-level-console",
        help="Logging level for console (e.g., DEBUG, INFO, WARNING)",
    ),
    log_level_file: str = typer.Option(
        "NOTSET",
        "--log-level-file",
        help="Logging level for file log (e.g., DEBUG, INFO, WARNING)",
    ),
):
    """Initialize logging before any command runs."""
    # Allow options to be parsed from anywhere in the command line
    ctx.resilient_parsing = True

    FORMAT = "%(message)s"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all logs, handlers will filter

    if log_level_file.upper() != "NOTSET":
        # File handler for DEBUG and above
        file_handler = logging.FileHandler("./logs/rr_trading.log")
        file_handler.setLevel(log_level_file.upper())
        file_handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(file_handler)

    # Console handler for INFO and above
    console_handler = RichHandler(markup=True)
    console_handler.setLevel(log_level_console.upper())
    console_handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(console_handler)

    # Remove default handlers if any (optional, but avoids duplicate logs)
    logger.propagate = False

    logger.debug(
        f"Initialized logger with console={log_level_console}, file={log_level_file}"
    )


@app.command(help="Print the current user configuration")
def print_config():
    """
    Print the current user configuration.
    """
    config = User.load()
    rprint(config)


@app.command(help="Run the setup wizard")
def setup():
    """
    Interactive setup wizard for the user configuration.
    """
    user = User.load()
    user.delete()
    etrade = Prompt.ask("Do you want to set up E-Trade Live?", default="n")
    if etrade.lower() == "y":
        # E-Trade Live setup
        user.etrade_live_secrets.set_api_key_from_file(
            Path(Prompt.ask("Enter your E-Trade API key path"))
        )
        user.etrade_live_secrets.set_api_secret_from_file(
            Path(Prompt.ask("Enter your E-Trade API secret path"))
        )
        user.save()

        etrade_authenticate = Prompt.ask(
            "Do you want to authenticate E-Trade now?", default="n"
        )
        if etrade_authenticate.lower() == "y":
            # E-Trade authentication
            from trading.cli.etrade.etrade import authenticate

            authenticate(False)
    etrade = Prompt.ask("Do you want to set up E-Trade Sandbox?", default="n")
    if etrade.lower() == "y":
        # E-Trade Sandbox setup
        user.etrade_sandbox_secrets.set_api_key_from_file(
            Path(Prompt.ask("Enter your E-Trade API key path"))
        )
        user.etrade_sandbox_secrets.set_api_secret_from_file(
            Path(Prompt.ask("Enter your E-Trade API secret path"))
        )
        user.save()

        etrade_authenticate = Prompt.ask(
            "Do you want to authenticate E-Trade now?", default="n"
        )
        if etrade_authenticate.lower() == "y":
            # E-Trade authentication
            from trading.cli.etrade.etrade import authenticate

            authenticate(True)
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


if __name__ == "__main__":
    app()
