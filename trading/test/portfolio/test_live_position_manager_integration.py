from trading.cli.alg.config import PortfolioConfig
from trading.src.portfolio.portfolio import Portfolio
from trading.src.portfolio.position import LivePositionManager


class DummyClient:
    def get_positions(self):
        return {"AAPL": []}


def test_portfolio_with_live_position_manager_initial_cash():
    pm = LivePositionManager(
        trading_client=DummyClient(), symbols=["AAPL"], initial_cash=1000.0
    )
    cfg = PortfolioConfig()
    pf = Portfolio(cfg, symbols=["AAPL"], position_manager=pm)

    assert pf.initial_cash == 1000.0
    assert pf.cash == 1000.0
    assert pf.net_value() == 1000.0
