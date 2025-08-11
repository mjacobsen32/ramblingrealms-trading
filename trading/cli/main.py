import logging
from pathlib import Path

import typer
from pydantic import SecretStr
from rich import print
from rich.prompt import Prompt

from trading.cli.alg import alg
from trading.cli.data import data
from trading.cli.etrade import etrade
from trading.cli.utils import init_logger
from trading.src.user_cache.user_cache import UserCache as User
from trading.src.utility.utils import read_key

app = typer.Typer(name="rr_trading", help="rr_trading CLI commands")
app.add_typer(etrade.app, name="etrade", help="E-Trade API commands")
app.add_typer(data.app, name="data", help="Data CLI commands")
app.add_typer(alg.app, name="alg", help="Algorithmic commands")


class AppState:
    def __init__(self):
        self.file_log_level = "INFO"
        self.console_log_level = "INFO"


FORMAT = "%(message)s"


@app.callback()
def main(
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
