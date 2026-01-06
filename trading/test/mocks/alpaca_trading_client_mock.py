import datetime
from typing import List, Optional
from uuid import UUID

from alpaca.broker import AccountStatus, AssetClass
from alpaca.trading import AssetExchange, PositionSide
from alpaca.trading.models import Calendar, Clock, Order, Position, TradeAccount
from alpaca.trading.requests import GetCalendarRequest, OrderRequest


class AlpacaTradingClientMock:
    def get_clock(self) -> Clock:
        return Clock(
            timestamp=datetime.datetime.fromisoformat("2023-01-01T10:00:00+00:00"),
            is_open=True,
            next_open=datetime.datetime.fromisoformat("2023-01-01T10:08:00+00:00"),
            next_close=datetime.datetime.fromisoformat("2023-01-01T10:16:00+00:00"),
        )

    def get_calendar(
        self,
        filters: Optional[GetCalendarRequest] = None,
    ) -> List[Calendar]:
        """Return a list of `Calendar` objects based on the provided filters.

        This implementation excludes weekends and uses the filters to determine
        the start and end dates. Each day's `open` is 09:30 UTC and `close` is 16:00 UTC.
        If the end date is before the start date, the function returns an empty list.
        """
        start = (
            filters.start if filters and filters.start else datetime.date(2012, 1, 1)
        )
        end = (
            filters.end
            if filters and filters.end
            else datetime.date.today() - datetime.timedelta(days=100)
        )

        if start > end:
            return []

        calendars: List[Calendar] = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Exclude weekends (Monday=0, Sunday=6)
                calendars.append(
                    Calendar(
                        date=current.strftime("%Y-%m-%d"), open="09:30", close="16:00"
                    )
                )
            current += datetime.timedelta(days=1)

        return calendars

    def submit_order(self, order_data: OrderRequest) -> Order:
        return Order(
            id=UUID("{12345678-1234-5678-1234-567812345678}"),
            client_order_id="12345678-1234-5678-1234-567812345678",
            created_at=datetime.datetime.fromisoformat("2023-01-01T10:00:00+00:00"),
            extended_hours=False,
            symbol=order_data.symbol,
            qty=order_data.qty,
            side=order_data.side,
            type=order_data.type,
            time_in_force=order_data.time_in_force,
            status="filled",
            filled_qty=order_data.qty,
            filled_avg_price=100.0,
            submitted_at=datetime.datetime.fromisoformat("2023-01-01T10:00:00+00:00"),
            updated_at=datetime.datetime.fromisoformat("2023-01-01T10:05:00+00:00"),
        )

    def get_account(self) -> TradeAccount:
        return TradeAccount(
            id=UUID("{12345678-1234-5678-1234-567812345678}"),
            account_number="account_number_123",
            status=AccountStatus.ACTIVE,
            cash="10000.0",
            buying_power="20000.0",
            portfolio_value="15000.0",
        )

    def get_all_positions(
        self,
    ) -> List[Position]:
        return [
            Position(
                symbol="AAPL",
                asset_id=UUID("{12345678-1234-5678-1234-567812345678}"),
                exchange=AssetExchange.NYSE,
                asset_class=AssetClass.US_EQUITY,
                side=PositionSide.LONG,
                cost_basis="1500.0",
                qty="10",
                avg_entry_price="150.0",
                current_price="155.0",
                market_value="1550.0",
                unrealized_pl="50.0",
                unrealized_plpc="0.0323",
            ),
            Position(
                symbol="TSLA",
                asset_id=UUID("{12345678-1234-5678-1234-567812345678}"),
                exchange=AssetExchange.NYSE,
                asset_class=AssetClass.US_EQUITY,
                side=PositionSide.LONG,
                cost_basis="3000.0",
                qty="5",
                avg_entry_price="600.0",
                current_price="620.0",
                market_value="3100.0",
                unrealized_pl="100.0",
                unrealized_plpc="0.0323",
            ),
        ]
