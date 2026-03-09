"""Tests for EDGAR data client."""

import json
from pathlib import Path
from unittest import mock

import pytest

from trading.src.data.edgar_client import EDGARClient


@pytest.fixture
def edgar_client(tmp_path):
    """Create EDGAR client with temporary cache directory."""
    return EDGARClient(cache_dir=tmp_path, cache_ttl_hours=24)


@pytest.fixture
def mock_ticker_map():
    """Mock SEC ticker-to-CIK mapping."""
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
    }


@pytest.fixture
def mock_company_facts():
    """Mock SEC company facts response."""
    return {
        "cik": 320193,
        "entityName": "Apple Inc",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "val": 400000000000,
                                "filed": "2025-02-01",
                            }
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "val": 100000000000,
                                "filed": "2025-02-01",
                            }
                        ]
                    }
                },
                "EarningsPerShareBasic": {
                    "units": {
                        "USD/shares": [
                            {
                                "end": "2024-12-31",
                                "val": 6.05,
                                "filed": "2025-02-01",
                            }
                        ]
                    }
                },
            }
        },
    }


def test_edgar_client_init(tmp_path):
    """Test EDGARClient initialization."""
    client = EDGARClient(cache_dir=tmp_path)
    assert client.cache_dir == tmp_path
    assert client.cache_ttl_hours == 24


def test_edgar_client_no_cache():
    """Test EDGARClient without caching."""
    client = EDGARClient(cache_dir=None)
    assert client.cache_dir is None


@mock.patch("trading.src.data.edgar_client.requests.Session.get")
def test_get_ticker_to_cik_map(mock_get, edgar_client, mock_ticker_map):
    """Test fetching and caching ticker-to-CIK mapping."""
    mock_response = mock.Mock()
    mock_response.json.return_value = mock_ticker_map
    mock_get.return_value = mock_response
    
    ticker_map = edgar_client.get_ticker_to_cik_map()
    
    assert "AAPL" in ticker_map
    assert ticker_map["AAPL"] == "0000320193"  # Zero-padded to 10 digits
    assert "MSFT" in ticker_map
    assert ticker_map["MSFT"] == "0000789019"


@mock.patch("trading.src.data.edgar_client.requests.Session.get")
def test_extract_financials(mock_get, edgar_client, mock_ticker_map, mock_company_facts):
    """Test extracting financial metrics from EDGAR data."""
    # Mock both calls: ticker map and company facts
    ticker_response = mock.Mock()
    ticker_response.json.return_value = mock_ticker_map
    
    facts_response = mock.Mock()
    facts_response.json.return_value = mock_company_facts
    
    mock_get.side_effect = [ticker_response, facts_response]
    
    financials = edgar_client.extract_financials("AAPL")
    
    assert financials is not None
    assert "edgar_revenue" in financials
    assert financials["edgar_revenue"] == 400000000000
    assert "edgar_net_income" in financials
    assert financials["edgar_net_income"] == 100000000000
    assert "edgar_eps" in financials
    assert financials["edgar_eps"] == 6.05


@mock.patch("trading.src.data.edgar_client.requests.Session.get")
def test_cache_hit(mock_get, edgar_client, mock_ticker_map):
    """Test that cache is used on subsequent calls."""
    mock_response = mock.Mock()
    mock_response.json.return_value = mock_ticker_map
    mock_get.return_value = mock_response
    
    # First call - should fetch
    edgar_client.get_ticker_to_cik_map()
    assert mock_get.call_count == 1
    
    # Second call - should use cache (no additional HTTP call)
    edgar_client.get_ticker_to_cik_map()
    assert mock_get.call_count == 1  # Still 1, not 2


def test_rate_limiting(edgar_client):
    """Test rate limiting between requests."""
    import time
    
    start = time.time()
    edgar_client._respect_rate_limit()
    edgar_client._respect_rate_limit()
    elapsed = time.time() - start
    
    # Should have slept at least 0.1 seconds (minimum interval)
    assert elapsed >= 0.09  # Small tolerance


@mock.patch("trading.src.data.edgar_client.requests.Session.get")
def test_ticker_not_found(mock_get, edgar_client, mock_ticker_map):
    """Test handling of unknown ticker."""
    mock_response = mock.Mock()
    mock_response.json.return_value = mock_ticker_map
    mock_get.return_value = mock_response
    
    edgar_client.get_ticker_to_cik_map()
    
    # Request for unknown ticker
    financials = edgar_client.extract_financials("UNKNOWN")
    
    assert financials is None
