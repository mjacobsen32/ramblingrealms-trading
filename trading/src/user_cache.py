import os
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from pathlib import Path
import json
from appdirs import user_config_dir
from rich.console import Console

class UserCache(BaseModel):
    """
    User Cache / Config for accessing and storing user-specific data.
    The cache contains any secrets, as well as the paths t secrets.
    The user may specify the JSON file to use for the cache via the environment variable
    `RR_TRADING_USER_CACHE_PATH`. If this variable is not set, the default path is
    `~/.rr_trading/user_cache.json`.
    """
    etrade_api_key_path: Path = Field(default="", description="E-Trade API key path")
    etrade_api_secret_path: Path = Field(default="", description="E-Trade API secret path")
    etrade_oauth_token: str = Field(default="", description="E-Trade OAuth Token. This field is set from E-Trade authentication")
    etrade_oauth_token_secret: str = Field(default="", description="E-Trade OAuth Token Secret. This field is set from E-Trade authentication")
    polygon_access_token_path: Path = Field(default="", description="Polygon Access Token Path")
    
    user_cache_path: Path = Path(os.environ.get("RR_TRADING_USER_CACHE_PATH", os.path.join(user_config_dir("rr_trading"), "user_cache.json")))

    model_config = ConfigDict(
        json_encoders={Path: str, None: str}
    )

    def __setattr__(self, name, value):
        """
        Override the __setattr__ method to save the cache whenever a new attribute is set."""
        super().__setattr__(name, value)
        self.save()

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if attr is None:
            Console().print(f"[bold red]Attribute {name} not found. Please run the setup command.")
        return attr

    @property
    def etrade_oauth_token(self) -> Optional[str]:
        """
        Get the E-Trade OAuth token.
        """
        if self.etrade_oauth_token :
            return self.etrade_oauth_token
        Console().print("[bold red]E-Trade OAuth token not found. Please authenticate first using the E-Trade CLI.")
        return None
    
    @property
    def etrade_oauth_token_secret(self) -> Optional[str]:
        """
        Get the E-Trade OAuth token secret.
        """
        if self.etrade_oauth_token :
            return self.etrade_oauth_token
        Console().print("[bold red]E-Trade OAuth token secret not found. Please authenticate first using the E-Trade CLI.")
        return None

    @property
    def etrade_api_key(self) -> str:
        """
        Get the E-Trade API key from the specified path.
        """
        if self.etrade_api_key_path and os.path.exists(self.etrade_api_key_path):
            return self.etrade_api_key_path.read_text(encoding="utf-8").strip()
        Console().print("[bold red]E-Trade API Key not found at {self.etrade_api_key_path}.")
        return "None"

    @property
    def etrade_api_secret(self) -> str:
        """
        Get the E-Trade API secret from the specified path.
        """
        if self.etrade_api_secret_path and os.path.exists(self.etrade_api_secret_path):
            return self.etrade_api_secret_path.read_text(encoding="utf-8").strip()
        Console().print("[bold red]E-Trade API Key Secret not found at {self.etrade_api_secret_path}.")
        return None
    
    @property
    def polygon_access_token(self) -> str:
        """
        Get the Polygon.io access token from the specified path.
        """
        if self.polygon_access_token_path and os.path.exists(self.polygon_access_token_path):
            return Path(self.polygon_access_token_path).read_text(encoding="utf-8").strip()
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