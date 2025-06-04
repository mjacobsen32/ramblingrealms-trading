import typer
from rich import print as rprint
from typing_extensions import Annotated
import json
from pathlib import Path

from trading.cli.alg.config import AlgConfig

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
):
    rprint("[blue]Starting training process...[/blue]")
    # Load configuration
    with Path.open(config) as f:
        config = AlgConfig.model_validate_json(f.read())
