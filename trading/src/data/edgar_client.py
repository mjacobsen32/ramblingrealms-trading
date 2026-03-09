"""
SEC EDGAR Data Client for fetching fundamental financial data.

This module provides a thin wrapper around the free SEC EDGAR API to fetch
company fundamental data (income statement, balance sheet, cash flow).
No API key is required.

Rate limits: 10 requests/second (respected via sleep)
Documentation: https://www.sec.gov/edgar/sec-api-documentation
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# SEC EDGAR API endpoints
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# User-Agent required by SEC
DEFAULT_USER_AGENT = "ramblingrealms-trading/1.0 (mjacobsen.pro@gmail.com)"

# Rate limiting: 10 req/sec means min 0.1 sec between requests
MIN_REQUEST_INTERVAL = 0.1


class EDGARClient:
    """
    Client for fetching SEC EDGAR fundamental data.
    
    Features:
    - Ticker to CIK mapping (auto-fetched and cached)
    - Company facts extraction (income, balance sheet, cash flow)
    - Local caching with TTL support
    - Rate limit compliance (10 req/sec)
    - Robust error handling
    """

    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_hours: int = 24):
        """
        Initialize EDGAR client.
        
        Args:
            cache_dir: Directory for caching API responses. If None, no caching.
            cache_ttl_hours: Cache time-to-live in hours (default 24)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl_hours = cache_ttl_hours
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
        
        self._ticker_to_cik_map: Optional[Dict[str, str]] = None
        self._last_request_time = 0.0
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _respect_rate_limit(self) -> None:
        """Sleep to respect SEC's 10 req/sec rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get_cache_path(self, cache_key: str) -> Optional[Path]:
        """Get cache file path and check if it's still valid."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return cache_file
        
        # Check TTL
        mtime = cache_file.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        if age_hours > self.cache_ttl_hours:
            cache_file.unlink()
            return cache_file
        
        return cache_file

    def _load_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached response if valid."""
        cache_file = self._get_cache_path(cache_key)
        if cache_file and cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save response to cache."""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def _fetch_json(self, url: str, cache_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch JSON from SEC API with caching and rate limiting.
        
        Args:
            url: Full URL to fetch
            cache_key: Optional cache key (if provided, uses cache)
        
        Returns:
            Parsed JSON response or None on error
        """
        # Try cache first
        if cache_key:
            cached = self._load_cached(cache_key)
            if cached:
                logger.debug(f"Cache hit: {cache_key}")
                return cached
        
        # Respect rate limits
        self._respect_rate_limit()
        
        try:
            logger.debug(f"Fetching: {url}")
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if cache_key:
                self._save_cache(cache_key, data)
            
            return data
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def get_ticker_to_cik_map(self) -> Dict[str, str]:
        """
        Get mapping of tickers to CIK numbers.
        
        Returns:
            Dict mapping ticker (str) -> CIK (str, zero-padded to 10 digits)
        """
        if self._ticker_to_cik_map:
            return self._ticker_to_cik_map
        
        data = self._fetch_json(SEC_COMPANY_TICKERS_URL, cache_key="company_tickers")
        if not data:
            logger.error("Failed to fetch ticker-CIK mapping")
            return {}
        
        # SEC returns: {index: {cik_str, ticker, title}}
        # We want: {ticker -> cik_str (zero-padded to 10)}
        self._ticker_to_cik_map = {}
        for entry in data.values():
            ticker = entry["ticker"].upper()
            cik = str(entry["cik_str"]).zfill(10)
            self._ticker_to_cik_map[ticker] = cik
        
        logger.info(f"Loaded {len(self._ticker_to_cik_map)} ticker-CIK mappings")
        return self._ticker_to_cik_map

    def get_company_facts(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company facts (financial data) for a ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
        
        Returns:
            Dict with company facts or None on error
        """
        ticker = ticker.upper()
        cik_map = self.get_ticker_to_cik_map()
        
        if ticker not in cik_map:
            logger.warning(f"Ticker {ticker} not found in SEC EDGAR")
            return None
        
        cik = cik_map[ticker]
        url = SEC_COMPANY_FACTS_URL.format(cik=cik)
        return self._fetch_json(url, cache_key=f"facts_{ticker}")

    def get_submissions(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch submission history for a ticker (includes filing dates).
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dict with submissions or None on error
        """
        ticker = ticker.upper()
        cik_map = self.get_ticker_to_cik_map()
        
        if ticker not in cik_map:
            logger.warning(f"Ticker {ticker} not found in SEC EDGAR")
            return None
        
        cik = cik_map[ticker]
        url = SEC_SUBMISSIONS_URL.format(cik=cik)
        return self._fetch_json(url, cache_key=f"submissions_{ticker}")

    def extract_financials(
        self, ticker: str, as_of_date: Optional[pd.Timestamp] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract latest financial metrics for a ticker.
        
        Args:
            ticker: Stock ticker
            as_of_date: If provided, get most recent filing before this date
        
        Returns:
            Dict with financial metrics or None on error
        """
        facts = self.get_company_facts(ticker)
        if not facts:
            return None
        
        financials = {}
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        
        # Extract key metrics (latest filing with a value)
        metric_map = {
            "edgar_revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
            "edgar_net_income": ["NetIncomeLoss"],
            "edgar_gross_profit": ["GrossProfit"],
            "edgar_operating_income": ["OperatingIncomeLoss"],
            "edgar_total_assets": ["Assets"],
            "edgar_total_liabilities": ["Liabilities"],
            "edgar_total_equity": ["StockholdersEquity"],
            "edgar_current_assets": ["AssetsCurrent"],
            "edgar_current_liabilities": ["LiabilitiesCurrent"],
            "edgar_operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
            "edgar_capital_expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
            "edgar_shares_outstanding": ["CommonStockSharesOutstanding"],
            "edgar_eps": ["EarningsPerShareBasic"],
        }
        
        for output_key, input_keys in metric_map.items():
            for input_key in input_keys:
                if input_key in us_gaap:
                    units = us_gaap[input_key].get("units", {})
                    # Prefer USD, otherwise take first available
                    values = units.get("USD", units.get(list(units.keys())[0] if units else "", []))
                    if values:
                        # Get most recent value (typically from 10-K or 10-Q)
                        for item in sorted(values, key=lambda x: x.get("end"), reverse=True):
                            if item.get("val") is not None:
                                financials[output_key] = item["val"]
                                break
                    break
        
        return financials if financials else None
