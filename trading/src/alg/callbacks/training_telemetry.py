"""
Enhanced telemetry callback for detailed training insights.
Tracks portfolio metrics, action distributions, and trading patterns.
"""

import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TradingTelemetryCallback(BaseCallback):
    """
    Custom callback for tracking detailed trading metrics during training.

    Logs:
    - Portfolio value and equity changes
    - Action statistics (buy/sell/hold distribution)
    - Episode-level metrics (win rate, drawdown)
    - Per-ticker performance insights
    """

    def __init__(
        self, verbose: int = 0, log_freq: int = 256, action_threshold: float = 0.1
    ):
        """
        Args:
            verbose: Verbosity level (0: silent, 1: info, 2: debug)
            log_freq: How often to log detailed metrics (in steps)
            action_threshold: Threshold to consider an action as buy/sell/hold
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.action_threshold = action_threshold

        # Episode tracking
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_equity_start: list[float] = []
        self.episode_equity_end: list[float] = []

        # Action statistics
        self.action_history: list[np.ndarray] = []
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        # Portfolio metrics
        self.max_portfolio_value = 0
        self.min_portfolio_value = float("inf")
        self.cumulative_trades = 0

        self.current_episode_start_value = None

    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        self.max_portfolio_value = 0
        self.min_portfolio_value = float("inf")
        logging.info("Training telemetry callback initialized")

    def _on_rollout_start(self) -> None:
        """Called at the beginning of a new rollout (before collecting data)."""
        pass

    def _on_step(self) -> bool:
        """
        Called after each step in the environment.

        Returns:
            bool: If False, training will be stopped.
        """
        # Get action from the last step
        if (
            self.model.ep_info_buffer is not None
            and len(self.model.ep_info_buffer) > 0
            and self.locals.get("actions") is not None
        ):
            actions = self.locals["actions"]

            # Track action statistics
            if isinstance(actions, np.ndarray):
                self.action_history.append(actions.copy())

                self.buy_count += np.sum(actions > self.action_threshold)
                self.sell_count += np.sum(actions < -self.action_threshold)
                self.hold_count += np.sum(np.abs(actions) <= self.action_threshold)

        # Extract info from environment
        infos = self.locals.get("infos", [{}])
        if len(infos) > 0 and isinstance(infos[0], dict):
            info = infos[0]

            # Track portfolio value if available
            if "net_value" in info:
                net_value = info["net_value"]
                self.max_portfolio_value = max(self.max_portfolio_value, net_value)
                self.min_portfolio_value = min(self.min_portfolio_value, net_value)

                # Track episode start value
                if self.current_episode_start_value is None:
                    self.current_episode_start_value = net_value

        # Track episode completion
        dones = self.locals.get("dones", [False])
        if any(dones):
            # Episode ended
            ep_info = {}
            if (
                self.model.ep_info_buffer is not None
                and len(self.model.ep_info_buffer) > 0
            ):
                ep_info = self.model.ep_info_buffer[-1]

            if "r" in ep_info:
                episode_reward = ep_info["r"]
                episode_length = ep_info.get("l", 0)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                # Calculate episode metrics
                if self.current_episode_start_value is not None and len(infos) > 0:
                    if "net_value" in infos[0]:
                        end_value = infos[0]["net_value"]
                        self.episode_equity_start.append(
                            self.current_episode_start_value
                        )
                        self.episode_equity_end.append(end_value)

                        # Log episode summary
                        equity_change = end_value - self.current_episode_start_value
                        equity_pct = (
                            equity_change / self.current_episode_start_value
                        ) * 100

                        self.logger.record(
                            "episode/equity_start", self.current_episode_start_value
                        )
                        self.logger.record("episode/equity_end", end_value)
                        self.logger.record("episode/equity_change", equity_change)
                        self.logger.record("episode/equity_change_pct", equity_pct)

                # Reset episode tracking
                self.current_episode_start_value = None

        # Periodic detailed logging
        if self.n_calls % self.log_freq == 0:
            self._log_detailed_metrics()

        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of a rollout (after collecting n_steps).
        This is when PPO does its policy update.
        """
        # Log rollout statistics
        if len(self.action_history) > 0:
            recent_actions = np.concatenate(self.action_history[-self.log_freq :])

            self.logger.record("rollout/action_mean", float(np.mean(recent_actions)))
            self.logger.record("rollout/action_std", float(np.std(recent_actions)))
            self.logger.record("rollout/action_max", float(np.max(recent_actions)))
            self.logger.record("rollout/action_min", float(np.min(recent_actions)))

            # Action distribution
            total_actions = self.buy_count + self.sell_count + self.hold_count
            if total_actions > 0:
                self.logger.record(
                    "actions/buy_pct", (self.buy_count / total_actions) * 100
                )
                self.logger.record(
                    "actions/sell_pct", (self.sell_count / total_actions) * 100
                )
                self.logger.record(
                    "actions/hold_pct", (self.hold_count / total_actions) * 100
                )

            # Reset counters
            self.buy_count = 0
            self.sell_count = 0
            self.hold_count = 0
            self.action_history = []

    def _log_detailed_metrics(self) -> None:
        """Log detailed metrics about training progress."""
        # Portfolio metrics
        if self.max_portfolio_value > 0:
            self.logger.record("portfolio/max_value", self.max_portfolio_value)
            self.logger.record("portfolio/min_value", self.min_portfolio_value)

            if self.min_portfolio_value > 0:
                drawdown = (
                    (self.max_portfolio_value - self.min_portfolio_value)
                    / self.max_portfolio_value
                    * 100
                )
                self.logger.record("portfolio/max_drawdown_pct", drawdown)

        # Episode statistics (last 100 episodes)
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]

            self.logger.record(
                "episode/mean_reward_100", float(np.mean(recent_rewards))
            )
            self.logger.record("episode/std_reward_100", float(np.std(recent_rewards)))
            self.logger.record(
                "episode/mean_length_100", float(np.mean(recent_lengths))
            )

            # Win rate (episodes with positive reward)
            win_rate = (np.array(recent_rewards) > 0).sum() / len(recent_rewards) * 100
            self.logger.record("episode/win_rate_pct", win_rate)

        # Equity tracking
        if len(self.episode_equity_start) > 0:
            recent_starts = self.episode_equity_start[-100:]
            recent_ends = self.episode_equity_end[-100:]

            equity_changes = np.array(recent_ends) - np.array(recent_starts)
            equity_pcts = (equity_changes / np.array(recent_starts)) * 100

            self.logger.record(
                "equity/mean_change_pct_100", float(np.mean(equity_pcts))
            )
            self.logger.record("equity/std_change_pct_100", float(np.std(equity_pcts)))
            self.logger.record("equity/best_episode_pct", float(np.max(equity_pcts)))
            self.logger.record("equity/worst_episode_pct", float(np.min(equity_pcts)))

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        logging.info("Training telemetry summary:")
        logging.info(f"  Total episodes: {len(self.episode_rewards)}")
        logging.info(f"  Mean episode reward: {np.mean(self.episode_rewards):.2f}")
        logging.info(f"  Max portfolio value: {self.max_portfolio_value:.2f}")
        logging.info(f"  Min portfolio value: {self.min_portfolio_value:.2f}")
        if len(self.episode_rewards) > 0:
            logging.info(f"  Mean episode reward: {np.mean(self.episode_rewards):.2f}")
        else:
            logging.info("  Mean episode reward: N/A (no episodes completed)")
