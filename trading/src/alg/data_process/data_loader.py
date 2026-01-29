import logging
import os
from typing import Any, ClassVar, Dict, List, Type

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading import Asset, GetAssetsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus

from trading.cli.alg.config import (
    DataConfig,
    DataRequests,
    DataSourceType,
    FeatureConfig,
)
from trading.src.features.utils import get_feature_cols
from trading.src.user_cache import user_cache


class DataSource:
    """
    Base class for all data sources in the Rambling Realms trading system.
    This class provides a common interface for fetching and processing data from various sources.
    Subclasses should implement the `get_data` method to fetch data from their respective sources.
    """

    def __init__(self, **kwargs):
        pass

    # Registry for subclasses
    _registry: ClassVar[Dict[DataSourceType, Type["DataSource"]]] = {}

    def get_data(
        self,
        fetch_data: bool,
        request: DataRequests,
        df: pd.DataFrame,
        cache_path: str,
        start_date: str,
        end_date: str,
        time_step_unit: TimeFrameUnit = TimeFrameUnit("Day"),
        cache_enabled: bool = True,
        time_step_period: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement get_data method")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        type_val = getattr(cls, "TYPE", None)
        if type_val is not None:
            cls._registry[type_val] = cls

    @classmethod
    def factory(cls, data: dict) -> "DataSource":
        """
        Factory method to create an instance of a DataSource subclass based on the provided data.
        The data dictionary must contain a 'source' key that matches one of the DataSourceType enum values.
        Raises ValueError if the 'source' key is missing or if the type is unknown.
        """

        datasource_type = data.get("source")
        if not datasource_type:
            raise ValueError("Missing 'type' field in feature data")
        datasource_type_enum = DataSourceType(datasource_type)
        subclass = cls._registry.get(datasource_type_enum)
        if not subclass:
            raise ValueError(f"Unknown DataSourceType type: {datasource_type_enum}")
        return subclass(**data)


class EdgarDataLoader(DataSource):
    """
    Edgar SEC Data Loader for fetching company financial data from SEC EDGAR filings.

    This loader fetches fundamental financial data (income statement, balance sheet,
    cash flow) and aligns it to the dates when the data was **released** (filing date),
    not the period it represents. This is critical for avoiding look-ahead bias.

    The data is forward-filled from each filing date until the next filing date,
    ensuring that the model only uses information that would have been available
    at each point in time.

    Key Financial Metrics Extracted:
    - Revenue and revenue growth
    - Net income and profit margins
    - Total assets, liabilities, and equity
    - Operating cash flow and free cash flow
    - Current ratio and debt-to-equity
    - EPS (Earnings Per Share)
    - Return on equity (ROE) and return on assets (ROA)
    """

    TYPE: ClassVar[DataSourceType] = DataSourceType.EDGAR

    # Define the financial metrics we want to extract
    FINANCIAL_METRICS: ClassVar[List[str]] = [
        # Income Statement
        "edgar_revenue",
        "edgar_net_income",
        "edgar_gross_profit",
        "edgar_operating_income",
        "edgar_ebitda",
        # Balance Sheet
        "edgar_total_assets",
        "edgar_total_liabilities",
        "edgar_total_equity",
        "edgar_current_assets",
        "edgar_current_liabilities",
        "edgar_cash_and_equivalents",
        "edgar_total_debt",
        # Cash Flow
        "edgar_operating_cash_flow",
        "edgar_investing_cash_flow",
        "edgar_financing_cash_flow",
        "edgar_free_cash_flow",
        "edgar_capital_expenditures",
        # Derived Ratios
        "edgar_profit_margin",
        "edgar_gross_margin",
        "edgar_operating_margin",
        "edgar_current_ratio",
        "edgar_debt_to_equity",
        "edgar_roe",  # Return on Equity
        "edgar_roa",  # Return on Assets
        "edgar_revenue_growth_yoy",
        "edgar_earnings_growth_yoy",
        # Shares and EPS
        "edgar_shares_outstanding",
        "edgar_eps",
        "edgar_book_value_per_share",
    ]

    def get_data(
        self,
        fetch_data: bool,
        request: DataRequests,
        df: pd.DataFrame,
        cache_path: str,
        start_date: str,
        end_date: str,
        time_step_unit: str = TimeFrameUnit.Day,
        cache_enabled: bool = True,
        time_step_period: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetches financial data from SEC EDGAR and aligns it to price data timestamps.

        The data is aligned by filing date (when the data was released) to avoid
        look-ahead bias. Financial metrics are forward-filled until the next filing.

        Args:
            fetch_data: If True, forces fresh data fetch from EDGAR
            request: DataRequests configuration specifying which tickers to fetch
            df: Input DataFrame with price data (must have MultiIndex [timestamp, symbol])
            cache_path: Path to cache the EDGAR data
            start_date: Start date for data range
            end_date: End date for data range
            time_step_unit: Time step unit (typically Day for EDGAR data)
            cache_enabled: Whether to use caching
            time_step_period: Period of time step
            **kwargs: Additional keyword arguments

        Returns:
            DataFrame with EDGAR financial metrics added as columns
        """
        if df.empty:
            logging.warning(
                "EdgarDataLoader received empty DataFrame. "
                "EDGAR data requires base price data with timestamp/symbol index. "
                "Returning empty DataFrame."
            )
            return df

        # Get the symbols from the input dataframe
        symbols = df.index.get_level_values("symbol").unique().tolist()

        # Override symbols if specified in request
        request_symbols = request.kwargs.get("symbol_or_symbols", None)
        if request_symbols and request_symbols != ["ALL"]:
            symbols = [s for s in request_symbols if s in symbols]

        if not symbols:
            logging.warning("No symbols found for EDGAR data loader.")
            return df

        logging.info(
            "EdgarDataLoader processing %d symbols: %s",
            len(symbols),
            symbols[:5] if len(symbols) > 5 else symbols,
        )

        cache_file = os.path.join(cache_path, f"{request.dataset_name}_edgar.parquet")

        # Try to load from cache if enabled
        edgar_cache_df = pd.DataFrame()
        if cache_enabled and os.path.exists(cache_file) and not fetch_data:
            try:
                edgar_cache_df = pd.read_parquet(cache_file)
                logging.info(
                    "Loaded EDGAR cache from %s with %d rows",
                    cache_file,
                    len(edgar_cache_df),
                )
            except Exception as e:
                logging.warning("Failed to load EDGAR cache: %s", e)
                edgar_cache_df = pd.DataFrame()

        # Determine which symbols need fetching
        cached_symbols = set()
        if not edgar_cache_df.empty:
            cached_symbols = set(
                edgar_cache_df.index.get_level_values("symbol").unique()
            )

        symbols_to_fetch = [s for s in symbols if s not in cached_symbols or fetch_data]

        # Fetch EDGAR data for symbols not in cache
        if symbols_to_fetch:
            logging.info(
                "Fetching EDGAR data for %d symbols: %s",
                len(symbols_to_fetch),
                symbols_to_fetch[:5] if len(symbols_to_fetch) > 5 else symbols_to_fetch,
            )

            new_edgar_data = self._fetch_edgar_data(
                symbols_to_fetch, start_date, end_date, **kwargs
            )

            if not new_edgar_data.empty:
                # Combine with cached data
                if not edgar_cache_df.empty and not fetch_data:
                    edgar_cache_df = pd.concat([edgar_cache_df, new_edgar_data])
                    edgar_cache_df = edgar_cache_df[
                        ~edgar_cache_df.index.duplicated(keep="last")
                    ]
                else:
                    edgar_cache_df = new_edgar_data

                # Save to cache
                if cache_enabled:
                    try:
                        os.makedirs(cache_path, exist_ok=True)
                        edgar_cache_df.to_parquet(cache_file)
                        logging.info("Saved EDGAR cache to %s", cache_file)
                    except Exception as e:
                        logging.warning("Failed to save EDGAR cache: %s", e)

        if edgar_cache_df.empty:
            logging.warning(
                "No EDGAR data available. Adding NaN columns for EDGAR features."
            )
            # Add empty columns for all EDGAR features
            for col in self.FINANCIAL_METRICS:
                df[col] = np.nan
            return df

        # Align EDGAR data to the price data timestamps
        result_df = self._align_edgar_to_timestamps(df, edgar_cache_df)

        return result_df

    def _fetch_edgar_data(
        self, symbols: List[str], start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        """
        Fetch EDGAR financial data for the given symbols.

        Uses the edgartools library to fetch data from SEC EDGAR.
        Data is indexed by filing_date (when it was released) and symbol.

        Args:
            symbols: List of ticker symbols to fetch
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            DataFrame with financial metrics indexed by (filing_date, symbol)
        """
        try:
            from edgar import Company, set_identity
        except ImportError:
            logging.error(
                "edgartools package not installed. "
                "Install with: pip install edgartools"
            )
            return pd.DataFrame()

        # Set identity for SEC compliance (required by SEC regulations)
        edgar_identity = kwargs.get("edgar_identity", None)
        if edgar_identity is None:
            # Try to get from user cache
            try:
                user = user_cache.UserCache().load()
                edgar_identity = getattr(user, "edgar_identity", None)
            except Exception:
                pass

        if edgar_identity is None:
            edgar_identity = os.environ.get("EDGAR_IDENTITY", "anonymous@example.com")

        try:
            set_identity(edgar_identity)
        except Exception as e:
            logging.warning("Failed to set EDGAR identity: %s", e)

        all_data = []
        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)

        for symbol in symbols:
            try:
                logging.debug("Fetching EDGAR data for %s", symbol)
                company = Company(symbol)

                if company is None or company.facts is None:
                    logging.warning("No EDGAR facts available for %s, skipping", symbol)
                    continue

                # Get financial filings with their data
                symbol_data = self._extract_financial_data(
                    company, symbol, requested_start, requested_end
                )

                if symbol_data:
                    all_data.extend(symbol_data)

            except Exception as e:
                logging.warning("Failed to fetch EDGAR data for %s: %s", symbol, e)
                continue

        if not all_data:
            return pd.DataFrame()

        # Create DataFrame from all data
        result_df = pd.DataFrame(all_data)

        # Ensure filing_date is datetime
        result_df["filing_date"] = pd.to_datetime(result_df["filing_date"])

        # Set MultiIndex
        result_df = result_df.set_index(["filing_date", "symbol"])
        result_df.sort_index(inplace=True)

        return result_df

    def _extract_financial_data(
        self,
        company: Any,
        symbol: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        """
        Extract financial data from a Company object for all available filings.

        This method extracts key financial metrics from 10-K and 10-Q filings,
        using the filing date as the timestamp to avoid look-ahead bias.

        Args:
            company: edgar Company object
            symbol: Ticker symbol
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            List of dictionaries with financial data for each filing
        """
        results = []

        try:
            # Get filings for annual (10-K) and quarterly (10-Q) reports
            filings = company.get_filings(form=["10-K", "10-Q"])

            if filings is None or len(filings) == 0:
                logging.debug("No 10-K/10-Q filings found for %s", symbol)
                return results

            # Process each filing
            for filing in filings:
                try:
                    filing_date = pd.to_datetime(filing.filing_date)

                    # Skip filings outside our date range
                    # We include filings slightly before start_date to have historical data
                    lookback_start = start_date - pd.Timedelta(days=365)
                    if filing_date < lookback_start or filing_date > end_date:
                        continue

                    # Extract financial data from this filing
                    filing_data = self._extract_filing_financials(
                        company, filing, symbol, filing_date
                    )

                    if filing_data:
                        results.append(filing_data)

                except Exception as e:
                    logging.debug("Failed to process filing for %s: %s", symbol, e)
                    continue

        except Exception as e:
            logging.warning("Failed to get filings for %s: %s", symbol, e)

        return results

    def _extract_filing_financials(
        self, company: Any, filing: Any, symbol: str, filing_date: pd.Timestamp
    ) -> Dict[str, Any] | None:
        """
        Extract financial metrics from a single filing.

        Uses the Company Facts API to get standardized financial data.

        Args:
            company: edgar Company object
            filing: Filing object
            symbol: Ticker symbol
            filing_date: Filing date

        Returns:
            Dictionary with financial metrics or None if extraction fails
        """
        try:
            data = {
                "filing_date": filing_date,
                "symbol": symbol,
                "form_type": filing.form,
            }

            # Try to get financials from Company Facts API (more reliable)
            facts = company.facts

            if facts is None:
                return None

            # Extract key metrics using the facts API
            # The facts API provides historical data with period information

            # Get the fiscal period end date from the filing
            fiscal_period_end = None
            try:
                fiscal_period_end = pd.to_datetime(
                    getattr(filing, "period_of_report", None)
                    or getattr(filing, "fiscal_period", None)
                )
            except Exception:
                fiscal_period_end = filing_date - pd.Timedelta(days=45)  # Approximate

            data["fiscal_period_end"] = fiscal_period_end

            # Helper function to safely query facts
            def get_fact_value(
                concept_name: str, fallback_concepts: List[str] = None
            ) -> float | None:
                """Query facts for a concept, trying fallbacks if needed."""
                concepts_to_try = [concept_name]
                if fallback_concepts:
                    concepts_to_try.extend(fallback_concepts)

                for concept in concepts_to_try:
                    try:
                        query_result = facts.query().by_concept(concept).execute()
                        if query_result:
                            # Get the most recent value before or at filing date
                            for fact in sorted(
                                query_result,
                                key=lambda x: getattr(x, "end", filing_date),
                                reverse=True,
                            ):
                                fact_end = getattr(fact, "end", None)
                                if fact_end and pd.to_datetime(fact_end) <= filing_date:
                                    return getattr(fact, "numeric_value", None)
                            # If no date match, just return the first value
                            if query_result:
                                return getattr(query_result[0], "numeric_value", None)
                    except Exception:
                        continue
                return None

            # Income Statement Metrics
            data["edgar_revenue"] = get_fact_value(
                "Revenue",
                [
                    "Revenues",
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet",
                    "TotalRevenue",
                ],
            )
            data["edgar_net_income"] = get_fact_value(
                "NetIncome",
                ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAttributableToParent"],
            )
            data["edgar_gross_profit"] = get_fact_value(
                "GrossProfit", ["GrossProfitLoss"]
            )
            data["edgar_operating_income"] = get_fact_value(
                "OperatingIncome", ["OperatingIncomeLoss", "IncomeLossFromOperations"]
            )

            # Balance Sheet Metrics
            data["edgar_total_assets"] = get_fact_value("Assets", ["TotalAssets"])
            data["edgar_total_liabilities"] = get_fact_value(
                "Liabilities", ["TotalLiabilities", "LiabilitiesAndStockholdersEquity"]
            )
            data["edgar_total_equity"] = get_fact_value(
                "StockholdersEquity",
                [
                    "Equity",
                    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                ],
            )
            data["edgar_current_assets"] = get_fact_value(
                "AssetsCurrent", ["CurrentAssets"]
            )
            data["edgar_current_liabilities"] = get_fact_value(
                "LiabilitiesCurrent", ["CurrentLiabilities"]
            )
            data["edgar_cash_and_equivalents"] = get_fact_value(
                "CashAndCashEquivalentsAtCarryingValue",
                [
                    "Cash",
                    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                ],
            )
            data["edgar_total_debt"] = get_fact_value(
                "LongTermDebt", ["DebtLongTerm", "LongTermDebtNoncurrent", "TotalDebt"]
            )

            # Cash Flow Metrics
            data["edgar_operating_cash_flow"] = get_fact_value(
                "NetCashProvidedByUsedInOperatingActivities",
                ["OperatingCashFlow", "CashFlowsFromOperatingActivities"],
            )
            data["edgar_investing_cash_flow"] = get_fact_value(
                "NetCashProvidedByUsedInInvestingActivities",
                ["InvestingCashFlow", "CashFlowsFromInvestingActivities"],
            )
            data["edgar_financing_cash_flow"] = get_fact_value(
                "NetCashProvidedByUsedInFinancingActivities",
                ["FinancingCashFlow", "CashFlowsFromFinancingActivities"],
            )
            data["edgar_capital_expenditures"] = get_fact_value(
                "PaymentsToAcquirePropertyPlantAndEquipment",
                ["CapitalExpenditures", "CapitalExpenditure"],
            )

            # Shares Outstanding
            data["edgar_shares_outstanding"] = get_fact_value(
                "CommonStockSharesOutstanding",
                ["SharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"],
            )

            # Calculate derived metrics
            self._calculate_derived_metrics(data)

            # Only return if we have at least some data
            has_data = any(
                data.get(col) is not None
                for col in self.FINANCIAL_METRICS
                if col in data
            )

            return data if has_data else None

        except Exception as e:
            logging.debug(
                "Failed to extract financials from filing for %s: %s", symbol, e
            )
            return None

    def _calculate_derived_metrics(self, data: Dict[str, Any]) -> None:
        """
        Calculate derived financial ratios from base metrics.

        Args:
            data: Dictionary with base financial metrics (modified in place)
        """

        def safe_divide(numerator, denominator):
            """Safely divide, returning None if invalid."""
            if numerator is None or denominator is None:
                return None
            if denominator == 0:
                return None
            return numerator / denominator

        # Profit Margins
        data["edgar_profit_margin"] = safe_divide(
            data.get("edgar_net_income"), data.get("edgar_revenue")
        )
        data["edgar_gross_margin"] = safe_divide(
            data.get("edgar_gross_profit"), data.get("edgar_revenue")
        )
        data["edgar_operating_margin"] = safe_divide(
            data.get("edgar_operating_income"), data.get("edgar_revenue")
        )

        # Liquidity and Leverage Ratios
        data["edgar_current_ratio"] = safe_divide(
            data.get("edgar_current_assets"), data.get("edgar_current_liabilities")
        )
        data["edgar_debt_to_equity"] = safe_divide(
            data.get("edgar_total_debt"), data.get("edgar_total_equity")
        )

        # Profitability Ratios
        data["edgar_roe"] = safe_divide(
            data.get("edgar_net_income"), data.get("edgar_total_equity")
        )
        data["edgar_roa"] = safe_divide(
            data.get("edgar_net_income"), data.get("edgar_total_assets")
        )

        # Per Share Metrics
        data["edgar_eps"] = safe_divide(
            data.get("edgar_net_income"), data.get("edgar_shares_outstanding")
        )
        data["edgar_book_value_per_share"] = safe_divide(
            data.get("edgar_total_equity"), data.get("edgar_shares_outstanding")
        )

        # Free Cash Flow
        capex = data.get("edgar_capital_expenditures")
        ocf = data.get("edgar_operating_cash_flow")
        if capex is not None and ocf is not None:
            # CapEx is typically negative in cash flow, so we add it
            data["edgar_free_cash_flow"] = ocf - abs(capex)
        else:
            data["edgar_free_cash_flow"] = None

        # EBITDA approximation (if we don't have it directly)
        if data.get("edgar_ebitda") is None:
            # Rough approximation: Operating Income + D&A
            # Since we may not have D&A, use operating income as proxy
            data["edgar_ebitda"] = data.get("edgar_operating_income")

        # Growth rates will be calculated in alignment step (need historical data)
        data["edgar_revenue_growth_yoy"] = None
        data["edgar_earnings_growth_yoy"] = None

    def _align_edgar_to_timestamps(
        self, price_df: pd.DataFrame, edgar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align EDGAR financial data to price data timestamps.

        For each timestamp in the price data, we use the most recent EDGAR filing
        that was available on or before that date. This ensures no look-ahead bias.

        Args:
            price_df: Price data with MultiIndex (timestamp, symbol)
            edgar_df: EDGAR data with MultiIndex (filing_date, symbol)

        Returns:
            Price data with EDGAR columns added
        """
        result_df = price_df.copy()

        # Initialize EDGAR columns with NaN
        for col in self.FINANCIAL_METRICS:
            if col not in result_df.columns:
                result_df[col] = np.nan

        # Process each symbol separately
        symbols = result_df.index.get_level_values("symbol").unique()

        for symbol in symbols:
            if symbol not in edgar_df.index.get_level_values("symbol").unique():
                logging.debug("No EDGAR data for symbol %s", symbol)
                continue

            # Get symbol-specific data
            symbol_edgar = edgar_df.xs(symbol, level="symbol")
            symbol_prices = result_df.xs(symbol, level="symbol")

            if symbol_edgar.empty:
                continue

            # Get the filing dates for this symbol
            filing_dates = symbol_edgar.index.sort_values()

            # For each timestamp in price data, find the most recent filing
            for col in self.FINANCIAL_METRICS:
                if col not in symbol_edgar.columns:
                    continue

                # Create a series with filing dates and values
                filing_series = symbol_edgar[col].sort_index()

                # Calculate YoY growth rates
                if col == "edgar_revenue_growth_yoy":
                    revenue_series = symbol_edgar.get("edgar_revenue")
                    if revenue_series is not None:
                        filing_series = revenue_series.pct_change(
                            periods=4
                        )  # 4 quarters for YoY
                elif col == "edgar_earnings_growth_yoy":
                    earnings_series = symbol_edgar.get("edgar_net_income")
                    if earnings_series is not None:
                        filing_series = earnings_series.pct_change(periods=4)

                # Forward fill from each filing date
                for ts in symbol_prices.index:
                    # Find the most recent filing before or on this timestamp
                    valid_filings = filing_dates[filing_dates <= ts]
                    if len(valid_filings) > 0:
                        most_recent_filing = valid_filings[-1]
                        value = filing_series.get(most_recent_filing)
                        if value is not None and not pd.isna(value):
                            result_df.loc[(ts, symbol), col] = value

        logging.info(
            "Aligned EDGAR data to %d price rows for %d symbols",
            len(result_df),
            len(symbols),
        )

        return result_df


class AlpacaDataLoader(DataSource):
    """
    ! TODO add columns to and from parquet

    """

    TYPE: ClassVar[DataSourceType] = DataSourceType.ALPACA

    def get_data(
        self,
        fetch_data: bool,
        request: DataRequests,
        df: pd.DataFrame,
        cache_path: str,
        start_date: str,
        end_date: str,
        time_step_unit: str = TimeFrameUnit.Day,
        cache_enabled: bool = True,
        time_step_period: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetches data from Alpaca and caches it locally.
        """
        alpaca_history_client: StockHistoricalDataClient | None = kwargs.get(
            "alpaca_history_client", None
        )
        alpaca_trading_client: TradingClient | None = kwargs.get(
            "alpaca_trading_client", None
        )

        if alpaca_history_client is None and fetch_data:
            user = user_cache.UserCache().load()
            alpaca_history_client = StockHistoricalDataClient(
                user.alpaca_api_key.get_secret_value(),
                user.alpaca_api_secret.get_secret_value(),
            )
        if alpaca_trading_client is None and fetch_data:
            user = user_cache.UserCache().load()
            alpaca_trading_client = TradingClient(
                user.alpaca_api_key.get_secret_value(),
                user.alpaca_api_secret.get_secret_value(),
            )

        cache_file = os.path.join(cache_path, request.dataset_name + ".parquet")

        def _coerce_ts(value, tz):
            ts = pd.to_datetime(value)
            if tz is None:
                return ts.tz_localize(None) if ts.tzinfo else ts
            if ts.tzinfo is None:
                return ts.tz_localize(tz)
            return ts.tz_convert(tz)

        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)

        cache_df = pd.DataFrame()
        used_cache = False
        cache_coverage = None

        if cache_enabled and os.path.exists(cache_file) and not fetch_data:
            cache_df = pd.read_parquet(cache_file)
            used_cache = True
            cache_ts_index = cache_df.index.get_level_values("timestamp")
            cache_tz = getattr(cache_ts_index, "tz", None)
            cache_start = _coerce_ts(cache_ts_index.min(), cache_tz)
            cache_end = _coerce_ts(cache_ts_index.max(), cache_tz)
            requested_start = _coerce_ts(requested_start, cache_tz)
            requested_end = _coerce_ts(requested_end, cache_tz)
            cache_coverage = (cache_start, cache_end)
            logging.info(
                "Loaded cache %s covering %s to %s (%d rows)",
                cache_file,
                cache_start,
                cache_end,
                len(cache_df),
            )
        elif not cache_enabled:
            logging.info(
                "Cache disabled; will fetch requested range %s to %s.",
                requested_start,
                requested_end,
            )
        elif fetch_data:
            logging.info(
                "Forced fetch requested; ignoring any existing cache for %s.",
                request.dataset_name,
            )
        else:
            logging.info(
                "Cache missing; will fetch requested range %s to %s.",
                requested_start,
                requested_end,
            )

        if request.kwargs.get("symbol_or_symbols") == ["ALL"]:
            if alpaca_trading_client is None:
                raise ValueError(
                    "alpaca_trading_client must be provided when symbol_or_symbols is ['ALL']"
                )
            all_assets = alpaca_trading_client.get_all_assets(
                GetAssetsRequest(
                    status=AssetStatus.ACTIVE,
                    asset_class=AssetClass.US_EQUITY,
                )
            )
            # Type narrow: get_all_assets returns list of Asset objects
            assets: list[str] = [
                asset.symbol  # type: ignore[union-attr]
                for asset in all_assets
                if hasattr(asset, "tradable") and asset.tradable  # type: ignore[union-attr]
            ]
            request.kwargs["symbol_or_symbols"] = assets
        assets = request.kwargs["symbol_or_symbols"]
        logging.info("Total number of symbols to consider: %d", len(assets))

        segments_to_fetch = []
        if not fetch_data and alpaca_history_client is None:
            # If we're not allowed to fetch and no client provided, we can only use what's in cache
            if not used_cache:
                logging.warning(
                    "No cache available and fetch_data=False with no client provided. Returning empty dataframe."
                )
        elif not used_cache:
            # No cache - fetch everything if we can
            segments_to_fetch.append((requested_start, requested_end))
        else:
            # We have cache - check if we need to fetch missing ranges
            if cache_coverage is None:
                raise ValueError("cache_coverage should not be None when using cache")
            cache_start, cache_end = cache_coverage
            if requested_start < cache_start:
                segments_to_fetch.append((requested_start, cache_start))
            if requested_end > cache_end:
                segments_to_fetch.append((cache_end, requested_end))
            if not segments_to_fetch:
                logging.info(
                    "Cache fully covers requested range %s to %s; no additional fetch needed.",
                    requested_start,
                    requested_end,
                )

        fetched_parts: list[pd.DataFrame] = []
        for seg_start, seg_end in segments_to_fetch:
            logging.info(
                "Fetching missing data for %s to %s (segment length %s).",
                seg_start,
                seg_end,
                seg_end - seg_start,
            )
            for start_idx in range(0, len(assets), 200):
                end_idx = min(start_idx + 200, len(assets))
                logging.info(
                    "Fetching symbols slice %d-%d for %s to %s...",
                    start_idx,
                    end_idx,
                    seg_start,
                    seg_end,
                )
                request.kwargs["symbol_or_symbols"] = assets[start_idx:end_idx]
                request_params = StockBarsRequest(
                    timeframe=TimeFrame(time_step_period, time_step_unit),
                    start=seg_start,
                    end=seg_end,
                    **request.kwargs,
                )
                if alpaca_history_client is None:
                    raise ValueError(
                        "alpaca_history_client must be provided when fetching data"
                    )
                bars = alpaca_history_client.get_stock_bars(request_params)
                if hasattr(bars, "df"):
                    fetched_parts.append(bars.df)
                elif isinstance(bars, dict) and "df" in bars:
                    fetched_parts.append(bars["df"])
                else:
                    try:
                        fetched_parts.append(pd.DataFrame(bars))
                    except Exception:
                        logging.error(
                            "Unknown bars return type from Alpaca client: %s",
                            type(bars),
                        )
                        raise

        new_data = pd.concat(fetched_parts) if fetched_parts else pd.DataFrame()
        if not new_data.empty:
            logging.info(
                "Fetched %d new rows spanning %s to %s for %d symbols.",
                len(new_data),
                new_data.index.get_level_values("timestamp").min(),
                new_data.index.get_level_values("timestamp").max(),
                new_data.index.get_level_values("symbol").nunique(),
            )
        elif segments_to_fetch:
            logging.info("No rows returned for requested missing segments.")

        dataset_df = pd.concat([cache_df, new_data]) if used_cache else new_data
        if not dataset_df.empty:
            dataset_df.sort_index(level=["timestamp", "symbol"], inplace=True)
            dataset_df = dataset_df[~dataset_df.index.duplicated(keep="last")]

        request.kwargs["symbol_or_symbols"] = assets

        if cache_enabled:
            if dataset_df.empty:
                logging.info("Cache not updated because dataset is empty.")
            elif not new_data.empty or not used_cache:
                dataset_df.to_parquet(cache_file)
                logging.info(
                    "Cache saved to %s covering %s to %s (%d rows).",
                    cache_file,
                    dataset_df.index.get_level_values("timestamp").min(),
                    dataset_df.index.get_level_values("timestamp").max(),
                    len(dataset_df),
                )
            else:
                logging.info("Cache already up-to-date; no new file written.")
        else:
            logging.info("Caching is disabled; not saving data to cache.")

        full_df = pd.concat([df, dataset_df]) if not dataset_df.empty else df
        if not full_df.empty:
            full_df.sort_index(level=["timestamp", "symbol"], inplace=True)
            full_df = full_df[~full_df.index.duplicated(keep="last")]
            ts_index = full_df.index.get_level_values("timestamp")
            target_tz = getattr(ts_index, "tz", None)
            req_start = _coerce_ts(requested_start, target_tz)
            req_end = _coerce_ts(requested_end, target_tz)
            filtered = full_df.loc[(ts_index >= req_start) & (ts_index <= req_end)]
        else:
            filtered = full_df

        if not filtered.empty:
            logging.info(
                "Returning data spanning %s to %s (%d rows, %d symbols).",
                filtered.index.get_level_values("timestamp").min(),
                filtered.index.get_level_values("timestamp").max(),
                len(filtered),
                filtered.index.get_level_values("symbol").nunique(),
            )
        else:
            logging.info(
                "No data available for requested range %s to %s.",
                requested_start,
                requested_end,
            )

        return filtered


class DataLoader:
    """
    DataLoader class for loading and processing data from various sources.
    It initializes with a DataConfig and FeatureConfig, fetches data from the specified sources,
    and applies the specified features to the data.

    ! todo no drop on on live
    """

    def __init__(
        self,
        data_config: DataConfig,
        feature_config: FeatureConfig,
        fetch_data: bool = False,
        **kwargs,
    ):
        """
        Initializes the DataLoader with the given configurations.
        """
        self.data_config = data_config
        self.feature_config = feature_config

        self.df = pd.DataFrame()

        for request in self.data_config.requests:
            data = DataSource.factory(request.model_dump()).get_data(
                fetch_data,
                request,
                self.df,
                str(data_config.cache_path),
                data_config.start_date,
                data_config.end_date,
                TimeFrameUnit(data_config.time_step_unit),
                data_config.cache_enabled,
                data_config.time_step_period,
                **kwargs,
            )
            for feature in [f for f in self.feature_config.features]:
                if feature.source == request.dataset_name or feature.source is None:
                    self.df = feature.to_df(self.df, data)
                else:
                    logging.warning(
                        "Skipping feature %s for source %s",
                        feature.name,
                        request.dataset_name,
                    )

        # ! if df is empty, raise error probably due to features having mis matched source names

        self.columns = self.df.columns
        self.features = get_feature_cols(features=self.feature_config.features)
        self.df.sort_index(level=["timestamp", "symbol"], inplace=True)
        self.df = self.df.reorder_levels(["timestamp", "symbol"])
        self.df.dropna()

        logging.info(
            "Data Successfully Loaded. Num Rows: %d, Num features: %d",
            len(self.df),
            len(self.features),
        )
        logging.debug(
            "\nCurrent columns: %s\nCurrent Features: %s",
            [f for f in self.columns],
            self.feature_config.features,
        )
        logging.debug(
            "Current tickers: %s",
            self.df.index.get_level_values("symbol").unique().tolist(),
        )

    @classmethod
    def data_info(cls, df: pd.DataFrame) -> str:
        """
        Returns a string representation of the dataframe.
        """
        return f"start_date: {df.index.get_level_values('timestamp').min()}, end_date: {df.index.get_level_values('timestamp').max()}, shape: {df.shape}"

    def get_train_test(self):
        """
        Splits the DataFrame into training and validation sets based on the validation split ratio.
        """

        unique_timestamps = sorted(self.df.index.get_level_values("timestamp").unique())
        split_idx = int(
            len(unique_timestamps) * (1 - self.data_config.validation_split)
        )
        train_timestamps = unique_timestamps[:split_idx]
        test_timestamps = unique_timestamps[split_idx:]
        train = self.df[
            self.df.index.get_level_values("timestamp").isin(train_timestamps)
        ]
        test = self.df[
            self.df.index.get_level_values("timestamp").isin(test_timestamps)
        ]
        logging.info(
            "Train data: %s\nTest data: %s", self.data_info(train), self.data_info(test)
        )

        return train, test

    def to_csv(self, path: str):
        """
        Saves the DataFrame to a CSV file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.df.to_csv(
            path,
            index=True,
        )
        logging.info("Data saved to %s", path)
