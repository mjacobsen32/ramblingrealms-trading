from typing import List

import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from trading.cli.alg.config import StockEnv
from trading.src.features import utils as feature_utils
from trading.src.features.generic_features import Feature


class TradingEnv:
    def __init__(self, data: pd.DataFrame, cfg: StockEnv, features: List[Feature]):
        self.unique_symbols = data["tic"].unique()

        self.stock_dimension = len(self.unique_symbols)
        # len(data.index("symbol").unique())
        self.num_stock_shares = [0] * self.stock_dimension
        self.tech_indicator_list = feature_utils.get_feature_cols(features=features)
        self.state_space = (
            1
            + 2 * self.stock_dimension
            + len(self.tech_indicator_list) * self.stock_dimension
        )

        if isinstance(cfg.sell_cost_pct, List):
            sell_cost = [float(x) for x in cfg.sell_cost_pct]
        else:
            sell_cost = [
                float(cfg.sell_cost_pct) for _ in range(0, self.stock_dimension)
            ]
        if isinstance(cfg.buy_cost_pct, List):
            buy_cost = [float(x) for x in cfg.buy_cost_pct]
        else:
            buy_cost = [float(cfg.buy_cost_pct) for _ in range(0, self.stock_dimension)]

        data["date"] = data["timestamp"]
        self.gym = StockTradingEnv(
            df=data,
            stock_dim=self.stock_dimension,
            hmax=cfg.hmax,
            initial_amount=cfg.starting_amount,
            num_stock_shares=self.num_stock_shares,
            buy_cost_pct=buy_cost,
            sell_cost_pct=sell_cost,
            reward_scaling=cfg.reward_scaling,
            state_space=self.state_space,
            action_space=self.stock_dimension,
            tech_indicator_list=self.tech_indicator_list,
            turbulence_threshold=cfg.turbulence_threshold,
            risk_indicator_col="turbulence",
            make_plots=True,
            print_verbosity=10,
            day=0,
            initial=True,
            previous_state=[],
            model_name="",
            mode="",
            iteration="",
        )
