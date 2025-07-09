import pandas as pd
import vectorbt as vbt


class BackTesting:
    def __init__(self, model, data, env):
        self.model = model
        self.data = data
        self.env = env
        self.total_rewards = 0
        self.records = []

    def run(self):
        obs, _ = self.env.reset()
        terminated, truncated = False, False
        total_reward = 0

        while not terminated and not truncated:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

        self.total_rewards = total_reward
        close_df = self.env.data.pivot(index="timestamp", columns="tic", values="close")
        size_df = self.env.data.pivot(index="timestamp", columns="tic", values="size")
        self.pf = vbt.Portfolio.from_orders(
            close=close_df, size=size_df, init_cash=self.env.initial_cash
        )

    def plot(self):
        for c in self.data.tic.unique():
            self.pf[c].plot().show()

    def stats(self):
        return self.pf.stats()
