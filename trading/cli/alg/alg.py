from pathlib import Path

import typer
from cli.alg.config import AlgConfig
from rich import print as rprint
from src.alg.data_process.data_loader import DataLoader
from src.alg.trainers.train import Trainer
from src.user_cache.user_cache import UserCache
from typing_extensions import Annotated

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
    trainer = Trainer(alg_config, UserCache().load(), data_loader)
    trainer.train()
    if not no_test:
        trainer.test()
