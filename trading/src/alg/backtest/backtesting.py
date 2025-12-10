import pandas as pd

from trading.cli.alg.config import BackTestConfig
from trading.src.portfolio.portfolio import Portfolio


class BackTesting:
    """
    Backtesting class for trading strategies using vectorbt.
    """

    def __init__(
        self,
        model,
        env,
        backtest_config: BackTestConfig,
        data: pd.DataFrame | None = None,
    ):
        self.model = model
        self.env = env
        if data is not None:
            self.env.init_data(data)
        self.backtest_config = backtest_config

    def run(self) -> Portfolio:
        """
        Run the backtest using the provided model and environment.
        """
        obs, _ = self.env.reset()
        terminated, truncated = False, False

        while not terminated and not truncated:
            action, _states = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)

        if self.backtest_config.save_results:
            self.env.pf.save(
                str(self.backtest_config.results_path.as_path()), df=self.env.data
            )

        return self.env.pf
