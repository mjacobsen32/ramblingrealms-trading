import logging
from typing import Callable


class BaseLRSchedule:
    def __init__(self, cfg: dict | float):
        if isinstance(cfg, float):
            self.initial_value = cfg
        elif isinstance(cfg, dict):
            self.initial_value = cfg.get("initial_value", 0.001)
        else:
            raise ValueError("Invalid configuration type")

    @classmethod
    def create(cls, cfg: dict | float) -> "BaseLRSchedule":
        if isinstance(cfg, dict) and cfg["type"] == "linear":
            return LinearLRSchedule(cfg)
        elif isinstance(cfg, float):
            return BaseLRSchedule(cfg)
        elif isinstance(cfg, dict):
            raise ValueError(f"Unknown learning rate schedule type: {cfg['type']}")
        return BaseLRSchedule(cfg)

    def func(self, progress_remaining: float) -> float:
        """
        Get the learning rate
        :param progress_remaining: (float)
        :return: (float)
        """
        return self.initial_value

    def __call__(self, progress_remaining: float) -> float:
        return self.func(progress_remaining)


class LinearLRSchedule(BaseLRSchedule):
    def func(self, progress_remaining: float) -> float:
        """
        Get the current learning rate depending on remaining progress.
        :param progress_remaining: (float)
        :return: (float)
        """
        lr = progress_remaining * self.initial_value
        logging.info("Learning Rate: %f", lr)
        return lr
