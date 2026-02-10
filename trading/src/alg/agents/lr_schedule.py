import math


class BaseLRSchedule:
    def __init__(self, cfg: dict | float):
        if isinstance(cfg, float):
            self.initial_value = cfg
        elif isinstance(cfg, dict):
            self.initial_value = cfg.get("initial_value", 0.001)
            self.final_value = cfg.get("final_value", 0.0)
        else:
            raise ValueError("Invalid configuration type")

    @classmethod
    def create(cls, cfg: dict | float) -> "BaseLRSchedule":
        if isinstance(cfg, float):
            return BaseLRSchedule(cfg)

        if isinstance(cfg, dict):
            schedule_type = cfg.get("type", "constant")
            if schedule_type == "linear":
                return LinearLRSchedule(cfg)
            elif schedule_type == "cosine":
                return CosineAnnealingLRSchedule(cfg)
            elif schedule_type == "exponential":
                return ExponentialLRSchedule(cfg)
            elif schedule_type == "cosine_warmup":
                return CosineWarmupLRSchedule(cfg)
            elif schedule_type == "polynomial":
                return PolynomialLRSchedule(cfg)
            elif schedule_type == "constant":
                return BaseLRSchedule(cfg)
            else:
                raise ValueError(
                    f"Unknown learning rate schedule type: {schedule_type}"
                )

        raise ValueError(f"Invalid learning rate configuration: {cfg}")

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
    """
    Linear learning rate decay from initial_value to final_value.
    Standard linear annealing schedule.
    """

    def func(self, progress_remaining: float) -> float:
        """
        Get the current learning rate depending on remaining progress.
        :param progress_remaining: (float) 1.0 at start, 0.0 at end
        :return: (float) learning rate
        """
        return (
            progress_remaining * self.initial_value
            + (1.0 - progress_remaining) * self.final_value
        )


class CosineAnnealingLRSchedule(BaseLRSchedule):
    """
    Cosine annealing learning rate schedule.
    Smooth decay following cosine curve - often works better than linear.
    Popular in modern deep learning (e.g., ResNet, Transformers).

    Formula: lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + cos(π * progress))
    """

    def func(self, progress_remaining: float) -> float:
        """
        Cosine annealing from initial_value to final_value.
        :param progress_remaining: (float) 1.0 at start, 0.0 at end
        :return: (float) learning rate
        """
        progress = 1.0 - progress_remaining
        return self.final_value + 0.5 * (self.initial_value - self.final_value) * (
            1.0 + math.cos(math.pi * progress)
        )


class ExponentialLRSchedule(BaseLRSchedule):
    """
    Exponential learning rate decay.
    Decays faster early, slower later.

    Formula: lr = initial_lr * decay_rate^progress
    """

    def __init__(self, cfg: dict | float):
        super().__init__(cfg)
        if isinstance(cfg, dict):
            # decay_rate determines how fast it decays (default 0.1 means 10% of initial at end)
            decay_rate = cfg.get("decay_rate", 0.1)
        else:
            decay_rate = 0.1

        if not (0.0 < decay_rate <= 1.0):
            raise ValueError(
                f"decay_rate must be in the interval (0, 1], got {decay_rate}"
            )

        self.decay_rate = decay_rate
    def func(self, progress_remaining: float) -> float:
        """
        Exponential decay from initial_value to initial_value * decay_rate.
        :param progress_remaining: (float) 1.0 at start, 0.0 at end
        :return: (float) learning rate
        """
        progress = 1.0 - progress_remaining
        return self.initial_value * (self.decay_rate**progress)


class CosineWarmupLRSchedule(BaseLRSchedule):
    """
    Cosine annealing with linear warmup.
    Starts from small LR, linearly increases to initial_value during warmup,
    then cosine annealing to final_value.

    Very popular in Transformer training (BERT, GPT, etc).
    """

    def __init__(self, cfg: dict | float):
        super().__init__(cfg)
        if isinstance(cfg, dict):
            # warmup_fraction: fraction of training to use for warmup (default 0.1 = 10%)
            self.warmup_fraction = cfg.get("warmup_fraction", 0.1)
        else:
            self.warmup_fraction = 0.1

        # Validate warmup_fraction to avoid division-by-zero and invalid progress values
        if not (0.0 < self.warmup_fraction < 1.0):
            raise ValueError(
                f"warmup_fraction must be in the open interval (0, 1), got {self.warmup_fraction!r}"
            )
    def func(self, progress_remaining: float) -> float:
        """
        Linear warmup followed by cosine annealing.
        :param progress_remaining: (float) 1.0 at start, 0.0 at end
        :return: (float) learning rate
        """
        progress = 1.0 - progress_remaining

        # Warmup phase
        if progress < self.warmup_fraction:
            return self.initial_value * (progress / self.warmup_fraction)

        # Cosine annealing phase
        cosine_progress = (progress - self.warmup_fraction) / (
            1.0 - self.warmup_fraction
        )
        return self.final_value + 0.5 * (self.initial_value - self.final_value) * (
            1.0 + math.cos(math.pi * cosine_progress)
        )


class PolynomialLRSchedule(BaseLRSchedule):
    """
    Polynomial learning rate decay.
    More gradual than exponential, more controlled than linear.

    Formula: lr = (initial_lr - final_lr) * (progress_remaining**power) + final_lr
    """

    def __init__(self, cfg: dict | float):
        super().__init__(cfg)
        if isinstance(cfg, dict):
            # power: controls decay curve (1.0 = linear, 2.0 = quadratic, etc.)
            self.power = cfg.get("power", 2.0)
        else:
            self.power = 2.0

        # Validate power early to avoid runtime errors like 0.0 ** negative.
        if not isinstance(self.power, (int, float)):
            raise TypeError(
                f"PolynomialLRSchedule 'power' must be a number (int or float), got {type(self.power).__name__}"
            )
        if self.power < 0:
            raise ValueError(
                f"PolynomialLRSchedule 'power' must be non-negative, got {self.power}"
            )
    def func(self, progress_remaining: float) -> float:
        """
        Polynomial decay from initial_value to final_value.
        :param progress_remaining: (float) 1.0 at start, 0.0 at end
        :return: (float) learning rate
        """
        return (self.initial_value - self.final_value) * (
            progress_remaining**self.power
        ) + self.final_value
