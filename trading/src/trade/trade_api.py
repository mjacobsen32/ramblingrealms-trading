import logging

from alpaca.data.timeframe import TimeFrameUnit

from trading.cli.trading.trade_config import BrokerType, PortfolioConfig, RRTradeConfig
from trading.src.alg.agents.agents import Agent
from trading.src.alg.data_process.data_loader import (
    DataLoader,
    StockHistoricalDataClient,
)
from trading.src.alg.environments.trading_environment import TradingEnv
from trading.src.features.utils import get_feature_cols
from trading.src.portfolio.portfolio import (
    Portfolio,
    ProjectPath,
    TradingClient,
)
from trading.src.user_cache.user_cache import UserCache


class Trade:
    def __init__(self, config: RRTradeConfig, live: bool):
        self.config = config

        self.model, self.meta_data = Agent.load_agent(config.model_path.as_path(), None)
        self.active_symbols = self.meta_data.get("symbols", [])
        self.active_features = self.meta_data.get("features", [])

        portfolio_config = (
            PortfolioConfig(**self.meta_data["env_config"]["portfolio_config"])
            if config.portfolio_config is None
            else config.portfolio_config
        )

        # TODO infer time_step
        self.pf = Portfolio(
            portfolio_config, self.active_symbols, (TimeFrameUnit.Day, 1)
        )

        logging.info(
            "Loaded model type: '%s' version: '%s'\nSymbols: %s\nFeatures: %s",
            self.meta_data.get("type", "Unknown"),
            self.meta_data.get("version", "Unknown"),
            self.active_symbols,
            self.active_features,
        )

        self.live = live
        self.trading_client = TradingClient.from_config(config, live)

        user_cache = UserCache().load()
        alpaca_api_key = user_cache.alpaca_api_key
        alpaca_api_secret = user_cache.alpaca_api_secret

        # Required for market calendar data
        self.trading_client = TradingClient.from_config(config, live)

        self.market_data_client = StockHistoricalDataClient(
            alpaca_api_key.get_secret_value(), alpaca_api_secret.get_secret_value()
        )

    @classmethod
    def from_config(cls, config: RRTradeConfig, live: bool = False) -> "Trade":
        if config.broker == BrokerType.ALPACA:
            return AlpacaTrade(config, live)
        elif config.broker == BrokerType.LOCAL:
            return LocalTrade(config, live)
        else:
            raise ValueError(f"Unsupported broker type: {config.broker}")

    def _load_data(self):
        clock = self.market_data_client.get_clock()
        calendar = self.market_data_client.get_calendar()
        window = [entry for entry in calendar if entry.date < clock.next_open.date()][
            -self.meta_data["env_config"]["lookback_window"] :
        ]
        start, end = window[0], window[-1]
        logging.info("Loading data for model trade: [%s, %s]", start, end)

        requests = [
            DataRequests(
                dataset_name="live",
                source=DataRequests.Source.Alpaca,
                endpoint="StockBarsRequest",
                kwargs={
                    "symbol_or_symbols": self.active_symbols,
                    "adjustment": "split",
                },
            )
        ]

        # TODO infer timeframe and period
        data_config = DataConfig(
            start_date=str(start),
            end_date=str(end),
            time_step_unit=TimeFrameUnit.Day,
            time_step_period=1,
            cache_path=ProjectPath(),
            cache_enabled=False,
            requests=requests,
            validation_split=0.0,
        )

        feature_config = FeatureConfig(
            features=self.meta_data["features"], fill_strategy="interpolate"
        )

        return DataLoader(
            data_config=data_config, feature_config=feature_config, fetch_data=True
        )

    def get_prices(self) -> np.ndarray:
        req = StockLatestTradeRequest(symbol_or_symbols=self.active_symbols)
        latest = self.alpaca_history_client.get_stock_latest_trade(req)
        prices = [trade.price for trade in latest.values()]
        return np.array(prices)

    def run_model(self):
        self.data_loader = self._load_data()
        prices = self.get_prices()
        obs = TradingEnv.observation(
            self.data_loader.df,
            self.trading_client.state(self.active_symbols),
            get_feature_cols(self.active_features),
            prices,
        )
        actions, _states = self.model.predict(obs)

        data = self.data_loader.df.copy()
        # TODO move to data_loader
        data.loc[:, "action"] = actions
        data.loc[:, "price"] = prices
        data.loc[:, "size"] = 0.0

        self.pf.step(data, True)
        self.trading_client.execute_trades(data.loc[:, ["size", "price"]])
