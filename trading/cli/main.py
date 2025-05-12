import typer
from trading.src.user_cache import UserCache as User
from trading.cli.etrade import etrade
from trading.cli.data import data
from rich import print
from rich.prompt import Prompt

app = typer.Typer(name="rr_trading", help="rr_trading CLI commands")
app.add_typer(etrade.app, name="etrade", help="E-Trade API commands")
app.add_typer(data.app, name="data", help="Data CLI commands")


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
    etrade = Prompt.ask("Do you want to set up E-Trade?", default="n")
    if etrade.lower() == "y":
        # E-Trade setup
        user.etrade_api_key_path = Prompt.ask("Enter your E-Trade API key path")
        user.etrade_api_secret_path = Prompt.ask("Enter your E-Trade API secret path")

        etrade_authenticate = Prompt.ask(
            "Do you want to authenticate E-Trade now?", default="n"
        )
        if etrade_authenticate.lower() == "y":
            # E-Trade authentication
            from trading.cli.etrade.etrade import authenticate

            authenticate()
    polygon = Prompt.ask("Do you want to set up Polygon?", default="n")
    if polygon.lower() == "y":
        # Polygon setup
        user.polygon_access_token_path = Prompt.ask(
            "Enter your Polygon access token path"
        )
