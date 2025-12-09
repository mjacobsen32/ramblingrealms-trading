import pandas as pd

from trading.cli.alg.config import PortfolioConfig
from trading.src.portfolio.portfolio import Portfolio
from trading.src.portfolio.position import Position, PositionManager


def test_portfolio_uses_existing_position_manager_cash_and_nav():
    class FakeClient:
        def get_positions(self):
            return {
                "AAPL": [
                    Position(
                        symbol="AAPL",
                        lot_size=2,
                        enter_price=10.0,
                        enter_date=pd.Timestamp("2024-01-01"),
                    )
                ]
            }

    pm = PositionManager.from_client(
        FakeClient(), symbols=["AAPL"], initial_cash=5_000.0
    )
    cfg = PortfolioConfig()
    pf = Portfolio(cfg, symbols=["AAPL"], position_manager=pm)

    # Prices for NAV computation
    pf.persistent_df = pd.DataFrame(
        {"symbol": ["AAPL"], "price": [12.0], "close": [12.0], "size": [0.0]}
    )

    assert pf.initial_cash == 5_000.0
    assert pf.cash == 5_000.0
    assert pf.net_value() == 5_024.0
