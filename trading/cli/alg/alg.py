from pathlib import Path

import typer
from rich import print as rprint
from typing_extensions import Annotated

from trading.cli.alg.config import AlgConfig
from trading.src.alg.agents.agents import Agent
from trading.src.alg.backtest.backtesting import BackTesting
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
    rprint("[blue]Starting training process...[/blue]")
    # Load configuration
    with Path.open(Path(config)) as f:
        alg_config = AlgConfig.model_validate_json(f.read())
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    trade_env = TradingEnv(
        data=data_loader.df,
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    trade_env.reset()

    rprint("[blue]Environment Initialized.[/blue]")

    model = Agent(alg_config.agent_config, trade_env)
    model.learn()
    model.save()

    if not no_test:
        backtest(config=config)


@app.command(help="Run backtesting on the trained model.")
def backtest(
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to the configuration file.")
    ],
):
    rprint("[blue]Starting backtesting process...[/blue]")
    # Load configuration
    with Path.open(Path(config)) as f:
        alg_config = AlgConfig.model_validate_json(f.read())
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    trade_env = TradingEnv(
        data=data_loader.get_train_test()[0],
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
        backtest=True,
    )
    trade_env.reset()

    rprint("[blue]Environment Initialized.[/blue]")
    model = Agent(config=alg_config.agent_config, env=trade_env, load=True)

    bt = BackTesting(model=model, data=data_loader.get_train_test()[1], env=trade_env)
    bt.run()
    print(bt.stats())
    bt.plot()
