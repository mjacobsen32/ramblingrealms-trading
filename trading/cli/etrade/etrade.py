import pyetrade
import typer
from pydantic import SecretStr
from rich import print as rprint

from trading.src.user_cache.user_cache import UserCache as user

app = typer.Typer(name="etrade", help="E-Trade API commands")


@app.command(help="Renew E-Trade OAuth tokens")
def renew_oauth(
    sandbox: bool = typer.Option(
        True,
        "--sandbox",
        "-s",
        help="Use the E-Trade sandbox environment for testing.",
    )
):
    """
    Renew E-Trade OAuth tokens
    """
    user_cfg = user.load()
    etrade_secrets = user_cfg.get_active_secrets(sandbox=sandbox)

    oauth = pyetrade.ETradeAccessManager(
        etrade_secrets.api_key.get_secret_value(),
        etrade_secrets.api_secret.get_secret_value(),
        etrade_secrets.oauth_token.get_secret_value(),
        etrade_secrets.oauth_token_secret.get_secret_value(),
    )
    if oauth.renew_access_token():
        rprint("[green]OAuth tokens renewed successfully!")
    else:
        rprint("[red]Failed to renew OAuth tokens. Please authenticate again.")


@app.command(help="Authenticate with E-Trade. Will set OAuth tokens")
def authenticate(
    sandbox: bool = typer.Option(
        True,
        "--sandbox",
        "-s",
        help="Use the E-Trade sandbox environment for testing.",
    )
):
    """
    Authenticate with E-Trade
    """
    user_cfg = user.load()
    etrade_secrets = user_cfg.get_active_secrets(sandbox=sandbox)

    oauth = pyetrade.ETradeOAuth(
        etrade_secrets.api_key.get_secret_value(),
        etrade_secrets.api_secret.get_secret_value(),
    )  # Use the printed URL
    rprint(
        "[yellow]Please visit the following URL to authenticate: {}".format(
            oauth.get_request_token()
        )
    )
    verifier_code = input("Enter verification code: ")
    tokens = oauth.get_access_token(verifier_code)
    rprint("[green]Authentication successful!")

    etrade_secrets.oauth_token = SecretStr(tokens["oauth_token"])
    etrade_secrets.oauth_token_secret = SecretStr(tokens["oauth_token_secret"])
    user_cfg.set_active_secrets(secrets=etrade_secrets, sandbox=sandbox)


@app.command(help="List accounts")
def list_accounts(
    sandbox: bool = typer.Option(
        True,
        "--sandbox",
        "-s",
        help="Use the E-Trade sandbox environment for testing.",
    )
):
    """
    List accounts
    """
    etrade_secrets = user.load().get_active_secrets(sandbox=sandbox)
    etrade_accounts = pyetrade.ETradeAccounts(
        etrade_secrets.api_key.get_secret_value(),
        etrade_secrets.api_secret.get_secret_value(),
        etrade_secrets.oauth_token.get_secret_value(),
        etrade_secrets.oauth_token_secret.get_secret_value(),
        dev=sandbox,
    )
    accounts = etrade_accounts.list_accounts()["AccountListResponse"]["Accounts"][
        "Account"
    ]
    for account in accounts:
        rprint(etrade_accounts.get_account_balance(account["accountIdKey"]))
