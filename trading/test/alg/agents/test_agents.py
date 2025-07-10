import numpy as np
import pytest

from trading.src.alg.agents.agents import Agent
from trading.test.alg.test_fixtures import *


def test_model_save_load(tmp_path, agent, agent_config, trade_env):
    assert agent is not None, "Agent should be initialized"
    agent.save()

    loaded_agent = Agent.load_agent(agent_config, trade_env)


def test_learn(agent):
    result = agent.learn(timesteps=1000)
    assert result is not None, "Learning should return a result"


def test_predict(agent, trade_env):
    obs, _ = trade_env.reset()
    action, _states = agent.predict(obs)
    obs, reward, terminated, truncated, info = trade_env.step(action)
    action, _states = agent.predict(obs)
    assert np.all(action is not 0.0), "Action should not be None"
