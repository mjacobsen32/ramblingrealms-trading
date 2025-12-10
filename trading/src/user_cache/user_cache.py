import json
import logging
import os
from pathlib import Path
from typing import Optional

from appdirs import user_config_dir
from pydantic import BaseModel, Field, SecretStr, field_serializer


class UserCache(BaseModel):
    """
    User Cache / Config for accessing and storing user-specific data.
    The cache contains any secrets, as well as the paths t secrets.
    The user may specify the JSON file to use for the cache via the environment variable
    `RR_TRADING_USER_CACHE_PATH`. If this variable is not set, the default path is
    `~/.rr_trading/user_cache.json`.
    """

    polygon_access_token_path: Path = Field(
        default=Path(""), description="Polygon Access Token Path"
    )
    alpaca_api_key_live: SecretStr = Field(
        default=SecretStr(""),
        description="Alpaca API Key for Live Trading",
    )
    alpaca_api_secret_live: SecretStr = Field(
        default=SecretStr(""),
        description="Alpaca API Secret for Live Trading",
    )
    alpaca_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Alpaca API Key for Paper Trading and Market Data",
    )
    alpaca_api_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Alpaca API Secret for Paper Trading and Market Data",
    )
    r2_access_key_id: SecretStr = Field(
        default=SecretStr(""), description="R2/S3 access key for remote trading"
    )
    r2_secret_access_key: SecretStr = Field(
        default=SecretStr(""),
        description="R2/S3 secret key for remote trading",
    )
    r2_endpoint_url: str = Field(
        default="",
        description="R2/S3 endpoint URL for remote trading client (optional).",
    )
    out_dir: Path = Field(
        default=Path(""),
        description="Output directory for saving results",
    )
    backtest_dir: Path = Field(
        default=Path(""),
        description="Directory for storing backtest results",
    )

    @classmethod
    def user_cache_path(cls) -> Path:
        return Path(
            os.environ.get(
                "RR_TRADING_USER_CACHE_PATH",
                os.path.join(user_config_dir("rr_trading"), "user_cache.json"),
            )
        )

    @field_serializer(
        "alpaca_api_key",
        "alpaca_api_secret",
        "r2_secret_access_key",
        "alpaca_api_key_live",
        "alpaca_api_secret_live",
        "r2_access_key_id",
        when_used="json",
    )
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
    def load(cls) -> "UserCache":
        """
        Load the user cache from a JSON file.
        If the file does not exist, create it with default values.
        """
        path = cls.user_cache_path()
        if not path.exists():
            config = cls()
            config.save()
            return config

        with path.open("r") as f:
            data = json.load(f)
        return cls(**data)

    def save(self) -> None:
        """
        Save the user cache to a JSON file.
        """
        path = self.user_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=4))

    def delete(self) -> None:
        """
        Delete the user cache file.
        """
        path = self.user_cache_path()
        if path.exists():
            os.remove(path)
