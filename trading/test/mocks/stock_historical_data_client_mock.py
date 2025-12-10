from typing import Union

from alpaca.common.types import RawData
from alpaca.data.models.trades import Trade
from alpaca.data.requests import StockLatestTradeRequest

from trading.cli.alg.config import Dict


class StockHistoricalDataClientMock:
    def get_stock_latest_trade(
        self, request_params: StockLatestTradeRequest
    ) -> Union[Dict[str, Trade], RawData]:
        ret = {}
        for symbol in request_params.symbol_or_symbols:
            trade = Trade(
                symbol=symbol,
                raw_data={
                    "t": "2023-01-01T10:00:00Z",
                    "p": 100.0,
                    "s": 10,
                },
            )
            ret[symbol] = trade
        return ret
