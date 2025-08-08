import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit


def read_key(path: str) -> str:
    """
    Read a key from a file.

    Args:
        path (str): The path to the file containing the key.

    Returns:
        str: The key read from the file.
    """
    try:
        with open(path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Key file not found at {path}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the key: {e}")


def time_frame_unit_to_pd_timedelta(
    time_step: tuple[TimeFrameUnit, int],
) -> pd.Timedelta:
    """
    Convert Alpaca TimeFrameUnit to vectorbt compatible string.
    """
    unit, count = time_step
    if unit == TimeFrameUnit.Minute:
        return pd.Timedelta(minutes=count)
    elif unit == TimeFrameUnit.Hour:
        return pd.Timedelta(hours=count)
    elif unit == TimeFrameUnit.Day:
        return pd.Timedelta(days=count)
    elif unit == TimeFrameUnit.Week:
        return pd.Timedelta(weeks=count)
    elif unit == TimeFrameUnit.Month:
        return pd.Timedelta(days=30 * count)
    else:
        raise ValueError(f"Unsupported time frame unit: {unit}")
