import logging
from pathlib import Path

import typer
from typing_extensions import Annotated

from trading.cli.alg.config import ProjectPath, RRConfig
from trading.cli.utils import init_file_logger
from trading.src.alg.agents.agents import Agent
from trading.src.alg.backtest.backtesting import BackTesting, Portfolio
from trading.src.alg.data_process.data_loader import DataLoader
from trading.src.alg.environments.trading_environment import TradingEnv
from trading.src.alg.environments.fast_training_env import FastTrainingEnv
from trading.src.alg.environments.stateful_trading_env import StatefulTradingEnv

app = typer.Typer(
    name="alg", help="Algorithm training, testing, and evaluation commands."
)


@app.command(help="")
def train(
    ctx: typer.Context,
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
        False, "--no-test", "-t", help="Run the backtesting suite via the new model"
    ),
    fetch_data: bool = typer.Option(
        False,
        "--fetch-data",
        "-f",
        help="Fetch the latest data before training. Do not use Cache.",
    ),
):
    logging.info("Starting training process...")
    # Load configuration
    with Path.open(Path(config)) as f:
        alg_config = RRConfig.model_validate_json(f.read())
        init_file_logger(ctx.obj.file_log_level, str(ProjectPath.OUT_DIR))
    data_loader = DataLoader(
        data_config=alg_config.data_config,
        feature_config=alg_config.feature_config,
        fetch_data=fetch_data,
    )
    
    # Use FastTrainingEnv for training (10x faster)
    train_env = FastTrainingEnv(
        data=data_loader.get_train_test()[0],
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
        time_step=(
            alg_config.data_config.time_step_unit,
            alg_config.data_config.time_step_period,
        ),
    )
    train_env.reset()

    logging.info("FastTrainingEnv Initialized for high-speed training.")

    model = Agent(alg_config.agent_config, train_env, alg_config.data_config)
    model.learn()

    if not dry_run:
        model.save()

    if not no_test:
        # Use StatefulTradingEnv for backtesting (accurate evaluation)
        test_env = StatefulTradingEnv(
            data=data_loader.get_train_test()[1],
            cfg=alg_config.stock_env,
            features=alg_config.feature_config.features,
            time_step=(
                alg_config.data_config.time_step_unit,
                alg_config.data_config.time_step_period,
            ),
        )
        test_env.reset()
        
        bt = BackTesting(
            model=model,
            env=test_env,
            backtest_config=alg_config.backtest_config,
            data=data_loader.get_train_test()[1],
        )
        pf = bt.run()
        pf.analysis(alg_config.backtest_config.analysis_config, test_env.data)
    ProjectPath.cache()
    logging.info("Training completed successfully.")


@app.command(help="Run backtesting on the trained model.")
def backtest(
    ctx: typer.Context,
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to the configuration file.")
    ],
    on_train: bool = typer.Option(
        False,
        "--on-train",
        help="Run backtesting on the training data instead of test data.",
    ),
):
    logging.info("Starting backtesting process...")
    # Load configuration
    with Path.open(Path(config)) as f:
        alg_config = RRConfig.model_validate_json(f.read())
        init_file_logger(ctx.obj.file_log_level, str(ProjectPath.OUT_DIR))
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    
    # Use StatefulTradingEnv for backtesting (accurate evaluation)
    test_env = StatefulTradingEnv(
        data=data_loader.get_train_test()[1],
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
        time_step=(
            alg_config.data_config.time_step_unit,
            alg_config.data_config.time_step_period,
        ),
    )
    test_env.reset()

    logging.info("StatefulTradingEnv Initialized for backtesting.")
    model = Agent(
        config=alg_config.agent_config,
        env=test_env,
        data_config=alg_config.data_config,
        load=True,
    )

    bt = BackTesting(
        model=model,
        env=test_env,
        backtest_config=alg_config.backtest_config,
        data=(
            data_loader.get_train_test()[1]
            if not on_train
            else data_loader.get_train_test()[0]
        ),
    )
    pf = bt.run()
    pf.analysis(alg_config.backtest_config.analysis_config, test_env.data)
    ProjectPath.cache()
    logging.info("Backtesting completed successfully.")


@app.command(help="Run analysis on the backtest results.")
def analysis(
    ctx: typer.Context,
    alg_config: Annotated[
        str, typer.Option("--config", "-c", help="Path to the configuration file.")
    ],
    out_dir: Annotated[
        str, typer.Option("--out_dir", "-o", help="Path to the root output directory.")
    ],
):
    logging.info("Starting analysis process...")
    with Path.open(Path(alg_config)) as f:
        ProjectPath.OUT_DIR = Path(out_dir).resolve()
        config = RRConfig.model_validate_json(f.read())
        init_file_logger(ctx.obj.file_log_level, str(ProjectPath.OUT_DIR))

    pf = Portfolio.load(
        config.stock_env.portfolio_config, config.backtest_config.results_path.as_path()
    )
    pf.analysis(config.backtest_config.analysis_config)

    logging.info("Analysis completed successfully.")
