import typer
import os
from pathlib import Path
from rich import print
from appdirs import user_config_dir
import tomli
import tomli_w
import pyetrade

app = typer.Typer()

CONFIG_DIR = user_config_dir("etrade")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.toml")

DEFAULT_CONFIG = {
    "keys": {
        "public_key_path": "~/.etrade/pub.key",
        "private_key_path": "~/.etrade/priv.key",
        "oauth_token": "",
        "oauth_token_secret": "", 
    },
    "profile": "default"
}

def load_config():
    """
    Load E-Trade API configuration
    """
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "wb") as f:
            f.write(tomli_w.dumps(DEFAULT_CONFIG).encode("utf-8"))
    with open(CONFIG_PATH, "rb") as f:
        return tomli.load(f)

def save_config(config):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "wb") as f:
        f.write(tomli_w.dumps(config).encode("utf-8"))

def get_tokens():
    """
    Get {public, private} tokens from the config
    """
    config = load_config()
    private_path = Path(config["keys"]["private_key_path"])
    if not private_path.exists():
        raise FileNotFoundError(f"Private Key file not found: {private_path}")
    priv = private_path.read_text(encoding="utf-8").strip()
    
    public_path = Path(config["keys"]["public_key_path"])
    if not public_path.exists():
        raise FileNotFoundError(f"Public Key file not found: {public_path}")
    pub = public_path.read_text(encoding="utf-8").strip()
    return pub, priv

def get_oauth_tokens():
    """
    Get {public, private} OAuth tokens from the config
    """
    config = load_config()
    return config["keys"]["oauth_token"], config["keys"]["oauth_token_secret"]

@app.command(help="Print configuration")
def print_config():
    """
    Print E-Trade API configuration
    """
    print(load_config())

@app.command(help="Set the private key path")
def private_key(key_path: str):
    """
    Set the private key path
    """
    config = load_config()
    config["keys"]["private_key_path"] = os.path.expanduser(key_path)
    save_config(config)

@app.command(help="Set the public key path")
def public_key(key_path: str):
    """
    Set the public key path
    """
    config = load_config()
    config["keys"]["public_key_path"] = os.path.expanduser(key_path)
    save_config(config)
    
@app.command(help="Set the profile")
def profile(profile_name: str):
    """
    Set the profile
    """
    config = load_config()
    config["profile"] = profile_name
    save_config(config)

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
    pub, priv, = get_tokens()
    oauth_tok, oauth_tok_secret = get_oauth_tokens()
    accounts = pyetrade.ETradeAccounts(
        pub,
        priv,
        oauth_tok,
        oauth_tok_secret
    )
    print(accounts.list_accounts())

    