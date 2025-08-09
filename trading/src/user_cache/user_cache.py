import json
import logging
import os
from pathlib import Path
from typing import Optional

from appdirs import user_config_dir
from pydantic import BaseModel, Field, SecretStr, field_serializer
from rich.console import Console

from trading.src.user_cache.etrade_secrets import ETradeSecrets


class UserCache(BaseModel):
    """
    User Cache / Config for accessing and storing user-specific data.
    The cache contains any secrets, as well as the paths t secrets.
    The user may specify the JSON file to use for the cache via the environment variable
    `RR_TRADING_USER_CACHE_PATH`. If this variable is not set, the default path is
    `~/.rr_trading/user_cache.json`.
    """

    etrade_sandbox_secrets: ETradeSecrets = Field(
        default_factory=lambda: ETradeSecrets(),
        description="E-Trade Sandbox Configuration",
    )
    etrade_live_secrets: ETradeSecrets = Field(
        default_factory=lambda: ETradeSecrets(),
        description="E-Trade Live Configuration",
    )

    polygon_access_token_path: Path = Field(
        default=Path(""), description="Polygon Access Token Path"
    )
    alpaca_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Alpaca API Key",
    )
    alpaca_api_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Alpaca API Secret",
    )
    out_dir: Path = Field(
        default=Path(""),
        description="Output directory for saving results",
    )
    backtest_dir: Path = Field(
        default=Path(""),
        description="Directory for storing backtest results",
    )

    user_cache_path: Path = Path(
        os.environ.get(
            "RR_TRADING_USER_CACHE_PATH",
            os.path.join(user_config_dir("rr_trading"), "user_cache.json"),
        )
    )

    @field_serializer("alpaca_api_key", "alpaca_api_secret", when_used="json")
    def dump_secret(self, v):
        return v.get_secret_value()

    def __setattr__(self, name, value):
        """
        Override the __setattr__ method to save the cache whenever a new attribute is set.
        """
        super().__setattr__(name, value)
        self.save()

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if attr is None:
            logging.warning(
                "Attribute %s not found. Please run the setup command.", name
            )
        return attr

    def get_active_secrets(self, sandbox: bool = True) -> ETradeSecrets:
        """
        Get the active E-Trade secrets based on the sandbox flag.
        If sandbox is True, return the sandbox secrets, otherwise return the live secrets.
        """
        if sandbox:
            return self.etrade_sandbox_secrets
        return self.etrade_live_secrets

    def set_active_secrets(self, secrets: ETradeSecrets, sandbox: bool = True) -> None:
        """
        Set the active E-Trade secrets based on the sandbox flag.
        If sandbox is True, set the sandbox secrets, otherwise set the live secrets.
        """
        if sandbox:
            self.etrade_sandbox_secrets = secrets
        else:
            self.etrade_live_secrets = secrets

    @property
    def polygon_access_token(self) -> Optional[str]:
        """
        Get the Polygon.io access token from the specified path.
        """
        if self.polygon_access_token_path and os.path.exists(
            self.polygon_access_token_path
        ):
            return (
                Path(self.polygon_access_token_path).read_text(encoding="utf-8").strip()
            )
        return None

    @classmethod
    def load(cls, path: Path = user_cache_path) -> "UserCache":
        """
        Load the user cache from a JSON file.
        If the file does not exist, create it with default values.
        """
        if not path.exists():
            config = cls()
            config.save(path)
            return config

        with path.open("r") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: Path = user_cache_path) -> None:
        """
        Save the user cache to a JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=4))

    def delete(self, path: Path = user_cache_path) -> None:
        """
        Delete the user cache file.
        """
        if path.exists():
            os.remove(path)
