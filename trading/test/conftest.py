from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).parent / "configs"


@pytest.fixture
def rr_trading_user_cache_path(monkeypatch):
    """
    Provide a temporary directory and set the environment variable
    `RR_TRADING_USER_CACHE_PATH` to that directory for the lifetime of the fixture.

    The environment variable is automatically restored by `monkeypatch` when the
    fixture scope ends.
    """
    user_cache = CONFIG_DIR / "user_cache.json"
    monkeypatch.setenv("RR_TRADING_USER_CACHE_PATH", str(user_cache))
    return user_cache
