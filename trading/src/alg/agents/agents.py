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

    @classmethod
    def make_agent(cls, config: AgentConfig, env: TradingEnv):
        algo = config.algo.lower()
        if algo not in AGENT_REGISTRY:
            raise ValueError(f"Unsupported algorithm: {algo}")
        AgentClass = AGENT_REGISTRY[algo]

        return AgentClass(env=env, **config.kwargs)

    def __init__(self, config: AgentConfig, env: TradingEnv):
        """Initializes the agent with the given configuration and environment.
        Args:
            config (AgentConfig): Configuration for the agent.
            env (TradingEnv): The trading environment in which the agent will operate.
        """
        self.config = config
        self.model = Agent.make_agent(config, env)

    def learn(self, timesteps: Optional[int] = None):
        return self.model.learn(
            total_timesteps=timesteps if timesteps else self.config.kwargs["n_steps"],
            progress_bar=True,
        )

    def predict(self, obs):
        return self.model.predict(obs, self.config.deterministic)

    def save(self, path: Optional[str] = None):
        self.model.save(path if path else self.config.save_path)
        return self.model

    def load(self, path: Optional[str] = None):
        self.model.load(path if path else self.config.save_path)
        return self.model
