from typing import List

from trading.src.features.generic_features import Feature


def get_feature_cols(features: list[Feature] | list[str]) -> list[str]:
    """
    Get the names of all enabled features from a list of Feature instances.
    """
    if len(features) > 0 and not all(isinstance(f, (Feature, str)) for f in features):
        raise ValueError("All features must be instances of Feature or str.")
    return [
        name
        for f in features
        if isinstance(f, Feature) and f.enabled
        for name in f.get_feature_names()
    ]


def min_window_size(features: list[Feature] | list[str]) -> int:
    """
    Get the minimum window size required by the enabled features.

    This returns the maximum 'period' attribute value among enabled features.
    If none of the enabled features define a valid period, returns 1.
    Allows, allows us to pull all required data for non-nan feature data.
    ! will fail on edge case where slow_period from macd is greater than other periods
    """
    max_period = 1
    for f in features:
        if not getattr(f, "enabled", False):
            continue
        if not hasattr(f, "period"):
            continue
        period = getattr(f, "period")
        try:
            period_int = int(period)
        except (TypeError, ValueError):
            continue
        if period_int > max_period:
            max_period = period_int
    return max_period
