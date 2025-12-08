import datetime
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC

from trading.cli.alg.config import AgentConfig, DataConfig, ProjectPath
from trading.src.alg.agents.lr_schedule import BaseLRSchedule
from trading.src.alg.environments.trading_environment import TradingEnv

AGENT_REGISTRY: dict[str, Any] = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "ddpg": DDPG,
    "sac": SAC,
}


class Agent:
    """
    Base class for all agents in the Rambling Realms trading system.
    This class provides a common interface and basic functionality for all agents.
    """

    # @TODO ideally we have better scopes on our dependency injection, data config is ingested for saving meta data
    def __init__(
        self,
        config: AgentConfig,
        env: TradingEnv,
        data_config: DataConfig | None = None,
        load: bool = False,
    ):
        """
        Initializes the agent with the given configuration and environment.
        Args:
            config (AgentConfig): Configuration for the agent.
            env (TradingEnv): The trading environment in which the agent will operate.
            load (bool): Whether to load an existing agent or create a new one.
        """
        self.config: AgentConfig = config
        self.meta_data: dict = {}
        self.env: TradingEnv = env
        if load:
            # When loading, we may not need a data_config; it's stored in meta_data
            self.model, self.meta_data = Agent.load_agent(config, self.env)
        else:
            self.model = Agent.make_agent(config=config, env=self.env)
            self.meta_data = {
                "type": config.algo,
                "version": ProjectPath.VERSION,
                "symbols": self.env.symbols,
                "features": [f.model_dump(mode="json") for f in self.env.features],
                "env_config": self.env.cfg.model_dump(mode="json"),
                "data_config": (
                    data_config.model_dump(mode="json") if data_config else {}
                ),
                "created_at": str(datetime.datetime.now()),
            }

    @classmethod
    def make_agent(
        cls, config: AgentConfig, env: TradingEnv
    ) -> A2C | DDPG | DQN | PPO | SAC:
        """
        Creates an agent based on the provided configuration and environment.
        """
        algo = config.algo.lower()
        if algo not in AGENT_REGISTRY:
            raise ValueError(f"Unsupported algorithm: {algo}")
        AgentClass = AGENT_REGISTRY[algo]

        lr = BaseLRSchedule.create(config.kwargs.get("learning_rate", 0.1))
        config.kwargs.pop("learning_rate", None)
        return AgentClass(
            env=env,
            learning_rate=lr,
            tensorboard_log=(
                ProjectPath.OUT_DIR / "tensorboard" if ProjectPath.OUT_DIR else None
            ),
            **config.kwargs,
        )

    @classmethod
    def load_agent(cls, config: AgentConfig | Path, env: TradingEnv | None):
        """
        Loads an agent and its meta_data from a saved zip file without extracting to disk.
        Returns:
            model, meta_data
        """
        # Determine path
        if isinstance(config, AgentConfig):
            zip_path = config.save_path.as_path()
            algo = config.algo.lower()
        else:
            zip_path = config
            # If config is str, we can't get algo, so will read from meta_data.json

        with zipfile.ZipFile(zip_path, "r") as zipf:
            # Load meta_data.json directly from zip
            with zipf.open("meta_data.json") as f:
                meta_data = json.load(f)

                algo = meta_data.get("type", None).lower()
                if algo not in AGENT_REGISTRY:
                    raise ValueError(
                        f"Unsupported or missing algorithm in meta_data: {algo}"
                    )

            # Load model.zip directly from zip into memory
            with zipf.open("model.zip") as model_file:
                # Stable Baselines3 expects a file path, so we need to write to a temporary file
                with tempfile.NamedTemporaryFile(
                    suffix=".zip", delete=False
                ) as tmp_model:
                    tmp_model.write(model_file.read())
                    tmp_model.flush()
                    model = AGENT_REGISTRY[algo].load(tmp_model.name, env=env)
                Path(tmp_model.name).unlink()  # Clean up temp file

        return model, meta_data

    def learn(self, timesteps: Optional[int] = None):
        logging.debug("Starting training for %s agent.", self.config.algo)
        ret = self.model.learn(
            total_timesteps=(
                self.config.total_timesteps if timesteps is None else timesteps
            ),
            progress_bar=True,
        )
        logging.debug("Training completed for %s agent.", self.config.algo)
        return ret

    def predict(self, obs):
        return self.model.predict(obs, self.config.deterministic)

    def save(self, path: Optional[str] = None):
        # Determine save path
        save_zip_path = Path(path if path else str(self.config.save_path))
        save_dir = save_zip_path.with_suffix("")

        # Create directory for saving
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model zip inside the directory
        model_zip_path = save_dir / "model.zip"
        self.model.save(str(model_zip_path))

        # Save meta_data as JSON
        meta_path = save_dir / "meta_data.json"
        with open(meta_path, "w") as f:
            json.dump(self.meta_data, f, indent=2)

        # Zip the directory contents into the final zip file
        with zipfile.ZipFile(save_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in save_dir.iterdir():
                zipf.write(file, arcname=file.name)

        # Optionally, clean up the directory
        for file in save_dir.iterdir():
            file.unlink()
        save_dir.rmdir()

        return self.model
