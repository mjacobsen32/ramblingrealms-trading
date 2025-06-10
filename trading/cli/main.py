from pathlib import Path

import typer
from cli.alg import alg
from cli.data import data
from cli.etrade import etrade
from pydantic import SecretStr
from rich import print
from rich.prompt import Prompt
from src.user_cache.user_cache import UserCache as User
from src.utility.utils import read_key

app = typer.Typer(name="rr_trading", help="rr_trading CLI commands")
app.add_typer(etrade.app, name="etrade", help="E-Trade API commands")
app.add_typer(data.app, name="data", help="Data CLI commands")
app.add_typer(alg.app, name="alg", help="Algorithmic commands")


@app.command(help="Print the current user configuration")
def print_config():
    """
    Print the current user configuration.
    """
    config = User.load()
    print(config)


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
            from cli.etrade.etrade import authenticate

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
            from cli.etrade.etrade import authenticate

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
