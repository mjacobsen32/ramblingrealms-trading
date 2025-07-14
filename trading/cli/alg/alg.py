import logging
from pathlib import Path

import typer
from rich import print as rprint
from typing_extensions import Annotated

from trading.cli.alg.config import RRConfig
from trading.src.alg.agents.agents import Agent
from trading.src.alg.backtest.backtesting import BackTesting, Portfolio
from trading.src.alg.data_process.data_loader import DataLoader
from trading.src.alg.environments.trading_environment import TradingEnv

app = typer.Typer(
    name="alg", help="Algorithm training, testing, and evaluation commands."
)


@app.command(help="")
def train(
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to the configuration file.")
    ],
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Run the training in dry run mode without saving results.",
    ),
    no_test: bool = typer.Option(
        False, "--no_test", "-t", help="Run the backtesting suite via the new model"
    ),
):
    logging.info("[blue]Starting training process...[/blue]")
    # Load configuration
    with Path.open(Path(config)) as f:
        alg_config = RRConfig.model_validate_json(f.read())
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    trade_env = TradingEnv(
        data=data_loader.get_train_test()[1],
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    trade_env.reset()

    logging.info("[blue]Environment Initialized.[/blue]")

    model = Agent(alg_config.agent_config, trade_env)
    model.learn()

    if not dry_run:
        model.save()

    if not no_test:
        bt = BackTesting(
            model,
            trade_env,
            alg_config.backtest_config,
        )
        pf = bt.run()
        logging.info(pf.stats())
        pf.plot()


@app.command(help="Run backtesting on the trained model.")
def backtest(
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to the configuration file.")
    ],
):
    logging.info("[blue]Starting backtesting process...[/blue]")
    # Load configuration
    with Path.open(Path(config)) as f:
        alg_config = RRConfig.model_validate_json(f.read())
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    trade_env = TradingEnv(
        data=data_loader.get_train_test()[1],
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
        backtest=True,
    )
    trade_env.reset()

    logging.info("[blue]Environment Initialized.[/blue]")
    model = Agent(config=alg_config.agent_config, env=trade_env, load=True)

    bt = BackTesting(
        model=model,
        env=trade_env,
        backtest_config=alg_config.backtest_config,
    )
    pf = bt.run()
    logging.info(pf.stats())
    pf.plot()


@app.command(help="Run analysis on the backtest results.")
def analysis(
    alg_config: Annotated[
        str, typer.Option("--config", "-c", help="Path to the configuration file.")
    ],
):
    logging.info("[blue]Starting analysis process...[/blue]")
    with Path.open(Path(alg_config)) as f:
        config = RRConfig.model_validate_json(f.read())
    pf = Portfolio.load(config.backtest_config.results_path.as_path())
    logging.info(pf.stats())
    logging.info(pf.orders())
    pf.plot()
