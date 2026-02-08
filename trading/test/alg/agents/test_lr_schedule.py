import pytest

from trading.src.alg.agents.lr_schedule import BaseLRSchedule


def test_base_lr_schedule():

    base: BaseLRSchedule = BaseLRSchedule.create(0.01)
    linear: BaseLRSchedule = BaseLRSchedule.create(
        {"type": "linear", "initial_value": 0.02}
    )

    assert base(progress_remaining=1.0) == 0.01
    assert base(progress_remaining=0.0) == 0.01

    assert linear(progress_remaining=1.0) == 0.02
    assert linear(progress_remaining=0.5) == 0.01
    assert linear(progress_remaining=0.0) == 0.0


def test_constant_schedule_from_float():
    """Test constant schedule created from float."""
    schedule = BaseLRSchedule.create(3e-4)

    assert schedule.func(1.0) == 3e-4
    assert schedule.func(0.5) == 3e-4
    assert schedule.func(0.0) == 3e-4


def test_constant_schedule_from_dict():
    """Test constant schedule created from dict."""
    schedule = BaseLRSchedule.create({"type": "constant", "initial_value": 5e-4})

    assert schedule.func(1.0) == 5e-4
    assert schedule.func(0.5) == 5e-4
    assert schedule.func(0.0) == 5e-4


def test_linear_schedule():
    """Test linear learning rate decay."""
    schedule = BaseLRSchedule.create(
        {"type": "linear", "initial_value": 1e-3, "final_value": 1e-5}
    )

    # At start
    assert abs(schedule.func(1.0) - 1e-3) < 1e-10

    # At middle
    expected_mid = 0.5 * 1e-3 + 0.5 * 1e-5
    assert abs(schedule.func(0.5) - expected_mid) < 1e-10

    # At end
    assert abs(schedule.func(0.0) - 1e-5) < 1e-10


def test_cosine_schedule():
    """Test cosine annealing schedule."""
    schedule = BaseLRSchedule.create(
        {"type": "cosine", "initial_value": 1e-3, "final_value": 1e-5}
    )

    # At start: should be at initial_value
    assert abs(schedule.func(1.0) - 1e-3) < 1e-10

    # At end: should be at final_value
    assert abs(schedule.func(0.0) - 1e-5) < 1e-10

    # At middle: should be between initial and final
    mid_lr = schedule.func(0.5)
    assert 1e-5 < mid_lr < 1e-3

    # Cosine should decay smoothly
    lr_25 = schedule.func(0.75)  # 25% progress
    lr_50 = schedule.func(0.5)  # 50% progress
    lr_75 = schedule.func(0.25)  # 75% progress

    assert lr_25 > lr_50 > lr_75


def test_exponential_schedule():
    """Test exponential decay schedule."""
    schedule = BaseLRSchedule.create(
        {"type": "exponential", "initial_value": 1e-3, "decay_rate": 0.1}
    )

    # At start
    assert abs(schedule.func(1.0) - 1e-3) < 1e-10

    # At end: should be initial * decay_rate
    expected_end = 1e-3 * 0.1
    assert abs(schedule.func(0.0) - expected_end) < 1e-10

    # Should decay faster early than linear
    linear_schedule = BaseLRSchedule.create(
        {"type": "linear", "initial_value": 1e-3, "final_value": 1e-4}
    )

    # At 25% progress, exponential should be lower (faster decay)
    assert schedule.func(0.75) < linear_schedule.func(0.75)


def test_cosine_warmup_schedule():
    """Test cosine with warmup schedule."""
    schedule = BaseLRSchedule.create(
        {
            "type": "cosine_warmup",
            "initial_value": 1e-3,
            "final_value": 1e-5,
            "warmup_fraction": 0.1,
        }
    )

    # At very start: should be near zero (warmup phase)
    assert schedule.func(1.0) < 1e-4

    # At end of warmup (90% remaining): should be at initial_value
    assert abs(schedule.func(0.9) - 1e-3) < 1e-6

    # At end: should be at final_value
    assert abs(schedule.func(0.0) - 1e-5) < 1e-10

    # During warmup, should be increasing
    lr_start = schedule.func(1.0)
    lr_mid_warmup = schedule.func(0.95)
    lr_end_warmup = schedule.func(0.9)
    assert lr_start < lr_mid_warmup < lr_end_warmup


def test_polynomial_schedule():
    """Test polynomial decay schedule."""
    # Quadratic (power=2)
    schedule = BaseLRSchedule.create(
        {"type": "polynomial", "initial_value": 1e-3, "final_value": 1e-5, "power": 2.0}
    )

    # At start
    assert abs(schedule.func(1.0) - 1e-3) < 1e-10

    # At end
    assert abs(schedule.func(0.0) - 1e-5) < 1e-10

    # At middle (power=2 means quadratic)
    # (1e-3 - 1e-5) * 0.5^2 + 1e-5
    expected_mid = (1e-3 - 1e-5) * 0.25 + 1e-5
    assert abs(schedule.func(0.5) - expected_mid) < 1e-10


def test_polynomial_power_variations():
    """Test different polynomial powers."""
    initial, final = 1e-3, 1e-5

    # Linear (power=1)
    linear = BaseLRSchedule.create(
        {
            "type": "polynomial",
            "initial_value": initial,
            "final_value": final,
            "power": 1.0,
        }
    )

    # Quadratic (power=2)
    quadratic = BaseLRSchedule.create(
        {
            "type": "polynomial",
            "initial_value": initial,
            "final_value": final,
            "power": 2.0,
        }
    )

    # Cubic (power=3)
    cubic = BaseLRSchedule.create(
        {
            "type": "polynomial",
            "initial_value": initial,
            "final_value": final,
            "power": 3.0,
        }
    )

    # At 50% progress, higher powers decay faster (smaller LR)
    assert cubic.func(0.5) < quadratic.func(0.5) < linear.func(0.5)


def test_schedule_monotonicity():
    """Test that all schedules monotonically decrease."""
    configs = [
        {"type": "linear", "initial_value": 1e-3, "final_value": 1e-5},
        {"type": "cosine", "initial_value": 1e-3, "final_value": 1e-5},
        {"type": "exponential", "initial_value": 1e-3, "decay_rate": 0.1},
        {
            "type": "polynomial",
            "initial_value": 1e-3,
            "final_value": 1e-5,
            "power": 2.0,
        },
    ]

    for cfg in configs:
        schedule = BaseLRSchedule.create(cfg)

        # Sample at different progress points
        progress_values = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
        lrs = [schedule.func(p) for p in progress_values]

        # Check monotonic decrease (or equal for constant)
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1], f"Schedule {cfg['type']} not monotonic"


def test_cosine_warmup_monotonicity():
    """Test cosine warmup increases then decreases."""
    schedule = BaseLRSchedule.create(
        {
            "type": "cosine_warmup",
            "initial_value": 1e-3,
            "final_value": 1e-5,
            "warmup_fraction": 0.1,
        }
    )

    # During warmup: should increase
    lr_start = schedule.func(1.0)
    lr_mid_warmup = schedule.func(0.95)
    lr_end_warmup = schedule.func(0.9)
    assert lr_start < lr_mid_warmup < lr_end_warmup

    # After warmup: should decrease
    lr_after_warmup = schedule.func(0.5)
    lr_late = schedule.func(0.1)
    assert lr_end_warmup > lr_after_warmup > lr_late


def test_invalid_schedule_type():
    """Test that invalid schedule type raises error."""
    with pytest.raises(ValueError):
        BaseLRSchedule.create({"type": "invalid_type"})


def test_callable_interface():
    """Test that schedules can be called directly."""
    schedule = BaseLRSchedule.create(
        {"type": "cosine", "initial_value": 1e-3, "final_value": 1e-5}
    )

    # Should be callable
    lr = schedule(0.5)
    assert isinstance(lr, float)
    assert lr == schedule.func(0.5)
