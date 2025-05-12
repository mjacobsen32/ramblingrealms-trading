import typer
import pyetrade
from trading.cli.etrade import config
from trading.cli.etrade.config import (
    get_tokens,
    get_oauth_tokens,
    load_config,
    save_config,
)

app = typer.Typer(name="etrade", help="E-Trade API commands")
app.add_typer(config.app, name="config")


@app.command(help="Authenticate with E-Trade. Will set OAuth tokens")
def authenticate():
    """
    Authenticate with E-Trade
    """
    pub, priv = get_tokens()
    oauth = pyetrade.ETradeOAuth(pub, priv)
    print(oauth.get_request_token())  # Use the printed URL
    verifier_code = input("Enter verification code: ")
    tokens = oauth.get_access_token(verifier_code)

    config = load_config()
    config["keys"]["oauth_token"] = tokens["oauth_token"]
    config["keys"]["oauth_token_secret"] = tokens["oauth_token_secret"]
    save_config(config)


@app.command(help="List accounts")
def list_accounts():
    """
    List accounts
    """
    (
        pub,
        priv,
    ) = get_tokens()
    oauth_tok, oauth_tok_secret = get_oauth_tokens()
    accounts = pyetrade.ETradeAccounts(pub, priv, oauth_tok, oauth_tok_secret)
    print(accounts.list_accounts())
