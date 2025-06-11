from typing import List

from trading.src.features.generic_features import Feature


def get_feature_cols(features: List[Feature]) -> List[str]:
    return [name for f in features if f.enabled for name in f.get_feature_names()]
