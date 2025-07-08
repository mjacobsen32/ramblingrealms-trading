from pathlib import Path

import typer
from rich import print as rprint
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
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
    train_env.reset()

    rprint("[blue]Environment Initialized.[/blue]")
    model = PPO(
        "MlpPolicy",
        DummyVecEnv([lambda: train_env]),
        verbose=1,
        tensorboard_log=alg_config.output_dir,
    ).learn(10_000)

    model.save(alg_config.save_path + alg_config.name + "_model")


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
    train_env = TradingEnv(
        data=data_loader.df,
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    train_env.reset()

    rprint("[blue]Environment Initialized.[/blue]")
    obs, _ = train_env.reset()
    done = False
    total_reward = 0

    model = PPO.load(alg_config.save_path + alg_config.name + "_model.zip", train_env)

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = train_env.step(action)
        total_reward += reward

    print(f"Final Portfolio Value: {train_env.total_assets:.2f}")
    import matplotlib.pyplot as plt

    plt.plot(train_env.asset_memory)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(alg_config.output_dir + "/portfolio_value.png")
