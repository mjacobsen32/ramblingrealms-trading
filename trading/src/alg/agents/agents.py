from typing import Optional

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC

from trading.cli.alg.config import AgentConfig
from trading.src.alg.environments.trading_environment import TradingEnv

AGENT_REGISTRY = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "ddpg": DDPG, "sac": SAC}


class Agent:
    """
    Base class for all agents in the Rambling Realms trading system.
    This class provides a common interface and basic functionality for all agents.
    """

    def __init__(self, config: AgentConfig, env: TradingEnv, load: bool = False):
        """
        Initializes the agent with the given configuration and environment.
        Args:
            config (AgentConfig): Configuration for the agent.
            env (TradingEnv): The trading environment in which the agent will operate.
            load (bool): Whether to load an existing agent or create a new one.
        """
        self.config = config
        if load:
            self.model = Agent.load_agent(config, env)
        else:
            self.model = Agent.make_agent(config, env)
        self.env = env

    @classmethod
    def make_agent(cls, config: AgentConfig, env: TradingEnv):
        """
        Creates an agent based on the provided configuration and environment.
        """
        algo = config.algo.lower()
        if algo not in AGENT_REGISTRY:
            raise ValueError(f"Unsupported algorithm: {algo}")
        AgentClass = AGENT_REGISTRY[algo]

        return AgentClass(env=env, **config.kwargs)

    @classmethod
    def load_agent(cls, config: AgentConfig, env: TradingEnv):
        """
        Loads an agent from a saved model file.
        """

        algo = config.algo.lower()
        if algo not in AGENT_REGISTRY:
            raise ValueError(f"Unsupported algorithm: {algo}")
        AgentClass = AGENT_REGISTRY[algo]

        return AgentClass.load(str(config.save_path), env=env)

    def learn(self, timesteps: Optional[int] = None):
        return self.model.learn(
            total_timesteps=timesteps if timesteps else self.config.kwargs["n_steps"],
            progress_bar=True,
        )

    def predict(self, obs):
        return self.model.predict(obs, self.config.deterministic)

    def save(self, path: Optional[str] = None):
        self.model.save(path if path else str(self.config.save_path))
        return self.model
