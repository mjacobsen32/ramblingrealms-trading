import pandas as pd
import pytest
from alpaca.data.timeframe import TimeFrameUnit

from trading.src.utility.utils import read_key, time_frame_unit_to_pd_timedelta


def test_read_key():
    # Create a temporary file with a sample key
    temp_file = "test_key.txt"
    with open(temp_file, "w") as f:
        f.write("sample_key")
    # Test reading the key
    key = read_key(temp_file)
    assert key == "sample_key", f"Expected 'sample_key', but got '{key}'"
    # Clean up
    import os

    os.remove(temp_file)


def test_read_key_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_key("non_existent_file.txt")


@pytest.mark.parametrize(
    "time_step, expected",
    [
        ((TimeFrameUnit.Minute, 5), pd.Timedelta(minutes=5)),
        ((TimeFrameUnit.Hour, 1), pd.Timedelta(hours=1)),
        ((TimeFrameUnit.Day, 1), pd.Timedelta(days=1)),
        ((TimeFrameUnit.Week, 1), pd.Timedelta(weeks=1)),
        ((TimeFrameUnit.Month, 1), pd.Timedelta(days=30)),
    ],
)
def test_time_frame_unit_to_pd_timedelta(time_step, expected):
    result = time_frame_unit_to_pd_timedelta(time_step)
    assert result == expected, f"Expected {expected}, but got {result}"
