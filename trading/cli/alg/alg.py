from pathlib import Path

import typer
from finrl.agents.stablebaselines3.models import DRLAgent
from rich import print as rprint
from stable_baselines3.common.logger import configure
from typing_extensions import Annotated

from trading.cli.alg.config import AlgConfig
from trading.src.alg.data_process.data_loader import DataLoader
from trading.src.alg.environments.trading_environment import TradingEnv
from trading.src.alg.trainers.train import Trainer
from trading.src.user_cache.user_cache import UserCache

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
    train_env = TradingEnv(
        data=data_loader.df,
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    agent = DRLAgent(env=train_env.gym)
    model_a2c = agent.get_model("a2c", model_kwargs={"device": "cpu"})

    tmp_path = alg_config.output_dir + "/a2c"
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    import stable_baselines3.common.logger as log

    # new_logger_a2c.set_level(log.DISABLED)
    # model_a2c.set_logger(new_logger_a2c)
    # print(new_logger_a2c.level)

    trained_a2c = agent.train_model(
        model=model_a2c, tb_log_name="a2c", total_timesteps=5000
    )
    # trainer = Trainer(alg_config, UserCache().load(), data_loader)
    # trainer.train()
    # if not no_test:
    # trainer.test()
