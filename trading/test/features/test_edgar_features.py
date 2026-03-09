"""Tests for EDGAR feature extraction."""

import numpy as np
import pandas as pd
import pytest

from trading.src.features.generic_features import (
    EdgarFundamentals,
    EdgarGrowth,
    EdgarLiquidity,
    EdgarProfitability,
    EdgarValuation,
    FeatureType,
)


@pytest.fixture
def sample_edgar_data():
    """Create sample DataFrame with EDGAR metrics."""
    dates = pd.date_range("2024-01-01", periods=5)
    symbols = ["AAPL", "MSFT"]
    index = pd.MultiIndex.from_product([dates, symbols], names=["timestamp", "symbol"])
    
    data = {
        "edgar_revenue": [400e9] * 10,
        "edgar_net_income": [100e9] * 10,
        "edgar_gross_profit": [150e9] * 10,
        "edgar_operating_income": [120e9] * 10,
        "edgar_total_assets": [350e9] * 10,
        "edgar_total_equity": [200e9] * 10,
        "edgar_current_assets": [100e9] * 10,
        "edgar_current_liabilities": [80e9] * 10,
        "edgar_operating_cash_flow": [110e9] * 10,
        "edgar_capital_expenditures": [20e9] * 10,
        "edgar_shares_outstanding": [16e9] * 10,
        "edgar_eps": [6.25] * 10,
        "edgar_book_value_per_share": [12.50] * 10,
        "edgar_profit_margin": [0.25] * 10,
        "edgar_gross_margin": [0.375] * 10,
        "edgar_operating_margin": [0.30] * 10,
        "edgar_current_ratio": [1.25] * 10,
        "edgar_debt_to_equity": [0.5] * 10,
        "edgar_roe": [0.50] * 10,  # Return on Equity
        "edgar_roa": [0.286] * 10,  # Return on Assets
        "edgar_free_cash_flow": [90e9] * 10,
        "edgar_revenue_growth_yoy": [0.10] * 10,  # 10% growth
        "edgar_earnings_growth_yoy": [0.15] * 10,  # 15% growth
    }
    
    return pd.DataFrame(data, index=index)


class TestEdgarFundamentals:
    """Test EdgarFundamentals feature."""
    
    def test_feature_creation(self):
        """Test feature instantiation."""
        feature = EdgarFundamentals(
            type=FeatureType.EDGAR_FUNDAMENTALS,
            name="fundamentals",
        )
        assert feature.type == FeatureType.EDGAR_FUNDAMENTALS
        assert feature.name == "fundamentals"
    
    def test_get_feature_names(self):
        """Test feature name generation."""
        feature = EdgarFundamentals(
            type=FeatureType.EDGAR_FUNDAMENTALS,
            name="fundamentals",
            include_revenue=True,
            include_income=True,
        )
        names = feature.get_feature_names()
        assert "fundamentals_revenue" in names
        assert "fundamentals_net_income" in names
    
    def test_to_df(self, sample_edgar_data):
        """Test feature DataFrame generation."""
        feature = EdgarFundamentals(
            type=FeatureType.EDGAR_FUNDAMENTALS,
            name="fundamentals",
            normalize=True,
        )
        result = feature.to_df(sample_edgar_data.copy(), sample_edgar_data)
        
        assert "fundamentals_revenue" in result.columns
        assert result["fundamentals_revenue"].notna().any()


class TestEdgarProfitability:
    """Test EdgarProfitability feature."""
    
    def test_feature_creation(self):
        """Test feature instantiation."""
        feature = EdgarProfitability(
            type=FeatureType.EDGAR_PROFITABILITY,
            name="profitability",
        )
        assert feature.type == FeatureType.EDGAR_PROFITABILITY
    
    def test_margin_extraction(self, sample_edgar_data):
        """Test profit margin extraction."""
        feature = EdgarProfitability(
            type=FeatureType.EDGAR_PROFITABILITY,
            name="profitability",
            include_margins=True,
            include_returns=False,
        )
        result = feature.to_df(sample_edgar_data.copy(), sample_edgar_data)
        
        assert "profitability_profit_margin" in result.columns
        assert (result["profitability_profit_margin"] == 0.25).all()
    
    def test_returns_extraction(self, sample_edgar_data):
        """Test return metrics (ROE, ROA)."""
        feature = EdgarProfitability(
            type=FeatureType.EDGAR_PROFITABILITY,
            name="profitability",
            include_margins=False,
            include_returns=True,
        )
        result = feature.to_df(sample_edgar_data.copy(), sample_edgar_data)
        
        assert "profitability_roe" in result.columns
        assert "profitability_roa" in result.columns


class TestEdgarLiquidity:
    """Test EdgarLiquidity feature."""
    
    def test_current_ratio(self, sample_edgar_data):
        """Test current ratio extraction."""
        feature = EdgarLiquidity(
            type=FeatureType.EDGAR_LIQUIDITY,
            name="liquidity",
            include_current_ratio=True,
            include_leverage=False,
        )
        result = feature.to_df(sample_edgar_data.copy(), sample_edgar_data)
        
        assert "liquidity_current_ratio" in result.columns
        # Log transform: ln(1.25) ≈ 0.223
        assert np.isclose(result["liquidity_current_ratio"].iloc[0], np.log(1.25), atol=0.01)
    
    def test_debt_to_equity(self, sample_edgar_data):
        """Test debt-to-equity ratio."""
        feature = EdgarLiquidity(
            type=FeatureType.EDGAR_LIQUIDITY,
            name="liquidity",
            include_current_ratio=False,
            include_leverage=True,
        )
        result = feature.to_df(sample_edgar_data.copy(), sample_edgar_data)
        
        assert "liquidity_debt_to_equity" in result.columns


class TestEdgarGrowth:
    """Test EdgarGrowth feature."""
    
    def test_revenue_growth(self, sample_edgar_data):
        """Test revenue growth extraction."""
        feature = EdgarGrowth(
            type=FeatureType.EDGAR_GROWTH,
            name="growth",
            include_revenue_growth=True,
            include_earnings_growth=False,
        )
        result = feature.to_df(sample_edgar_data.copy(), sample_edgar_data)
        
        assert "growth_revenue_growth" in result.columns
        assert (result["growth_revenue_growth"] == 0.10).all()
    
    def test_clip_outliers(self):
        """Test clipping of extreme growth values."""
        dates = pd.date_range("2024-01-01", periods=3)
        symbols = ["AAPL"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["timestamp", "symbol"])
        
        data = {
            "edgar_revenue_growth_yoy": [10.0, -10.0, 2.0],  # Extreme values
            "edgar_earnings_growth_yoy": [5.0, -5.0, 1.0],
        }
        df = pd.DataFrame(data, index=index)
        
        feature = EdgarGrowth(
            type=FeatureType.EDGAR_GROWTH,
            name="growth",
            clip_outliers=True,
            clip_range=(-5.0, 5.0),
        )
        result = feature.to_df(df.copy(), df)
        
        # Value 10.0 should be clipped to 5.0
        assert result["growth_revenue_growth"].iloc[0] == 5.0


class TestEdgarValuation:
    """Test EdgarValuation feature."""
    
    def test_eps_extraction(self, sample_edgar_data):
        """Test EPS extraction."""
        feature = EdgarValuation(
            type=FeatureType.EDGAR_VALUATION,
            name="valuation",
            include_eps=True,
            include_book_value=False,
        )
        result = feature.to_df(sample_edgar_data.copy(), sample_edgar_data)
        
        assert "valuation_eps" in result.columns
    
    def test_book_value_extraction(self, sample_edgar_data):
        """Test book value per share extraction."""
        feature = EdgarValuation(
            type=FeatureType.EDGAR_VALUATION,
            name="valuation",
            include_eps=False,
            include_book_value=True,
        )
        result = feature.to_df(sample_edgar_data.copy(), sample_edgar_data)
        
        assert "valuation_book_value" in result.columns


class TestEdgarFeaturesIntegration:
    """Integration tests for EDGAR features."""
    
    def test_all_features_together(self, sample_edgar_data):
        """Test using all EDGAR features together."""
        features = [
            EdgarFundamentals(type=FeatureType.EDGAR_FUNDAMENTALS, name="fundamentals"),
            EdgarProfitability(type=FeatureType.EDGAR_PROFITABILITY, name="profitability"),
            EdgarLiquidity(type=FeatureType.EDGAR_LIQUIDITY, name="liquidity"),
            EdgarGrowth(type=FeatureType.EDGAR_GROWTH, name="growth"),
            EdgarValuation(type=FeatureType.EDGAR_VALUATION, name="valuation"),
        ]
        
        df = sample_edgar_data.copy()
        for feature in features:
            df = feature.to_df(df, sample_edgar_data)
        
        # Check that we have features from all classes
        assert any("fundamentals" in col for col in df.columns)
        assert any("profitability" in col for col in df.columns)
        assert any("liquidity" in col for col in df.columns)
        assert any("growth" in col for col in df.columns)
        assert any("valuation" in col for col in df.columns)
    
    def test_missing_columns(self):
        """Test handling of missing EDGAR columns."""
        dates = pd.date_range("2024-01-01", periods=2)
        symbols = ["AAPL"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["timestamp", "symbol"])
        
        # Partial EDGAR data (missing some columns)
        df = pd.DataFrame({
            "other_col": [1, 2],
        }, index=index)
        
        feature = EdgarFundamentals(
            type=FeatureType.EDGAR_FUNDAMENTALS,
            name="fundamentals",
        )
        result = feature.to_df(df, df)
        
        # Features should be present even if source columns are missing
        assert len(result) == len(df)
