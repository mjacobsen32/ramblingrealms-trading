import os
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_serializer


class ETradeSecrets(BaseModel):
    """
    Secrets for E-Trade API.
    This class is used to store the E-Trade API key and secret, as well as the OAuth tokens.
    The paths to the keys and secrets are specified in the user cache.
    """

    api_secret: SecretStr = Field(
        default=SecretStr(""), description="E-Trade API Secret"
    )
    api_key: SecretStr = Field(default=SecretStr(""), description="E-Trade API")
    oauth_token: SecretStr = Field(
        default=SecretStr(""), description="E-Trade OAuth Token"
    )
    oauth_token_secret: SecretStr = Field(
        default=SecretStr(""), description="E-Trade OAuth Token Secret"
    )

    @field_serializer(
        "api_secret", "api_key", "oauth_token", "oauth_token_secret", when_used="json"
    )
    def dump_secret(self, v):
        return v.get_secret_value()

    def set_api_key_from_file(self, file_path: Path):
        """
        Read the E-Trade secrets from a JSON file.
        """
        if os.path.exists(file_path):
            self.api_key = SecretStr(file_path.read_text(encoding="utf-8").strip())
        else:
            raise FileNotFoundError(f"E-Trade API Key not found at {file_path}.")

    def set_api_secret_from_file(self, file_path: Path):
        """
        Read the E-Trade secrets from a JSON file.
        """
        if os.path.exists(file_path):
            self.api_secret = SecretStr(file_path.read_text(encoding="utf-8").strip())
        else:
            raise FileNotFoundError(f"E-Trade API Secret not found at {file_path}.")
