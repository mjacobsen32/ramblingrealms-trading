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
