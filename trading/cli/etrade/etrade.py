import typer
import pyetrade
from rich import print
from trading.src.user_cache import UserCache as user

app = typer.Typer(name="etrade", help="E-Trade API commands")


@app.command(help="Authenticate with E-Trade. Will set OAuth tokens")
def authenticate():
    """
    Authenticate with E-Trade
    """
    config = user.load()
    oauth = pyetrade.ETradeOAuth(config.etrade_api_key, config.etrade_api_secret)
    print(oauth.get_request_token())  # Use the printed URL
    verifier_code = input("Enter verification code: ")
    tokens = oauth.get_access_token(verifier_code)
    print("[green]Authentication successful!")

    config.etrade_oauth_token = tokens["oauth_token"]
    config.etrade_oauth_token_secret = tokens["oauth_token_secret"]


@app.command(help="List accounts")
def list_accounts():
    """
    List accounts
    """
    config = user.load()
    accounts = pyetrade.ETradeAccounts(
        config.etrade_api_key,
        config.etrade_api_secret,
        config.etrade_oauth_token,
        config.etrade_oauth_token_secret,
    )
    print(accounts.list_accounts())
