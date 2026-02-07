import numpy as np
import pandas as pd
import pytest
from gymnasium import Env

from trading.cli.alg.config import Path, StockEnv
from trading.src.alg.environments.fast_training_env import FastTrainingEnv
from trading.src.alg.environments.stateful_trading_env import StatefulTradingEnv
from trading.src.alg.environments.trading_environment import TradingEnv
from trading.src.features.generic_features import Feature
from trading.test.trade.test_trade_clients_integration import CONFIG_DIR

CONFIG_DIR = Path(__file__).parent.parent / "configs"
# Each step increments the observation index, so ROUNDS needs to be predetermined so we can load that much pseudo data in
ROUNDS = 1000


def benchmark_train_env(env: Env, actions: np.ndarray):
    env.step(action=actions)


class EnvBenchmarkArgs:
    def __init__(self, num_tickers, lookback_window, num_features, gpu=False):
        self.num_tickers = num_tickers
        self.lookback_window = lookback_window
        self.num_features = num_features
        self.gpu = gpu

    @property
    def data(self):
        # Create a dummy DataFrame with the specified dimensions
        index = pd.MultiIndex.from_product(
            [
                range(self.lookback_window + 1 + ROUNDS),
                [f"Ticker_{i}" for i in range(self.num_tickers)],
            ],
            names=["timestamp", "symbol"],
        )
        columns = self.features
        data = pd.DataFrame(
            np.random.rand(len(index), len(columns)), index=index, columns=columns
        )
        return data

    @property
    def features(self) -> list[str] | list[Feature]:
        ret: list[str] = ["price"]
        for i in range(len(ret), self.num_features):
            ret.append(f"Feature_{i}")
        return ret

    @property
    def stock_cfg(self) -> StockEnv:
        with Path.open(Path(CONFIG_DIR / "stock_env.json")) as f:
            stock_cfg = StockEnv.model_validate_json(f.read())
        stock_cfg.lookback_window = self.lookback_window
        return stock_cfg


def init_env(bm_args: EnvBenchmarkArgs) -> FastTrainingEnv:
    from trading.src.alg.environments.trading_environment import TradingEnv

    env = FastTrainingEnv(
        data=bm_args.data, cfg=bm_args.stock_cfg, features=bm_args.features
    )
    return env


def params() -> tuple[list[EnvBenchmarkArgs], list[str]]:
    ret: list[EnvBenchmarkArgs] = []
    ret_names: list[str] = []
    for num_tickers in [1, 100, 1000]:
        for lookback_window in [10, 100]:
            for num_features in [10, 1000]:
                for gpu in [False]:
                    ret.append(
                        EnvBenchmarkArgs(
                            num_tickers=num_tickers,
                            lookback_window=lookback_window,
                            num_features=num_features,
                            gpu=gpu,
                        )
                    )
                    ret_names.append(
                        f"{num_tickers},{lookback_window},{num_features},{gpu})"
                    )
    return ret, ret_names


def params_tickers() -> tuple[list[EnvBenchmarkArgs], list[str]]:
    ret: list[EnvBenchmarkArgs] = []
    ret_names: list[str] = []
    for num_tickers in [1, 10, 100, 1000, 10000]:
        ret.append(
            EnvBenchmarkArgs(
                num_tickers=num_tickers,
                lookback_window=10,
                num_features=10,
                gpu=False,
            )
        )
        ret_names.append(f"tickers:{num_tickers}")
    return ret, ret_names


def params_numfeatures() -> tuple[list[EnvBenchmarkArgs], list[str]]:
    ret: list[EnvBenchmarkArgs] = []
    ret_names: list[str] = []
    for num_features in [1, 10, 100, 1000, 10000, 100000]:
        ret.append(
            EnvBenchmarkArgs(
                num_tickers=1,
                lookback_window=10,
                num_features=num_features,
                gpu=False,
            )
        )
        ret_names.append(f"features:{num_features}")
    return ret, ret_names


def params_lb() -> tuple[list[EnvBenchmarkArgs], list[str]]:
    ret: list[EnvBenchmarkArgs] = []
    ret_names: list[str] = []
    for lookback_window in [10, 100, 1000]:
        ret.append(
            EnvBenchmarkArgs(
                num_tickers=1,
                lookback_window=lookback_window,
                num_features=10,
                gpu=False,
            )
        )
        ret_names.append(f"lookback:{lookback_window}")
    return ret, ret_names


@pytest.mark.benchmark
@pytest.mark.parametrize("args", params_lb()[0], ids=params_lb()[1])
def test_benchmark_lookback(benchmark, args):
    actions = np.random.rand(args.num_tickers)
    env = init_env(args)
    benchmark.pedantic(benchmark_train_env, args=(env, actions), rounds=ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("args", params_tickers()[0], ids=params_tickers()[1])
def test_benchmark_tickers(benchmark, args):
    actions = np.random.rand(args.num_tickers)
    env = init_env(args)
    benchmark.pedantic(benchmark_train_env, args=(env, actions), rounds=ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("args", params_numfeatures()[0], ids=params_numfeatures()[1])
def test_benchmark_numfeatures(benchmark, args):
    actions = np.random.rand(args.num_tickers)
    env = init_env(args)
    benchmark.pedantic(benchmark_train_env, args=(env, actions), rounds=ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("args", params()[0], ids=params()[1])
def test_benchmark_train(benchmark, args):
    actions = np.random.rand(args.num_tickers)
    env = init_env(args)
    benchmark.pedantic(benchmark_train_env, args=(env, actions), rounds=ROUNDS)
