import pytest

from trading.src.features.generic_features import *
from trading.src.features.generic_features import FillStrategy, OperationType


@pytest.fixture
def sin_wave_time_series():
    import matplotlib.pyplot as plt
    import pandas as pd

    n = 100
    amplitude = 1.0
    offset = 2.0

    x = np.linspace(0, 2 * np.pi, n)
    close_values = amplitude * np.sin(x) + offset
    high_values = close_values + 0.1 * np.random.randn(n)
    low_values = close_values - 0.1 * np.random.randn(n)
    volume_values = np.random.randint(1, 100, n)

    return pd.DataFrame(
        {
            "close": close_values,
            "high": high_values,
            "low": low_values,
            "volume": volume_values,
        }
    )


@pytest.fixture
def moving_average():
    return MovingWindow(
        type=FeatureType.MOVING_WINDOW,
        name="rolling_mean_50",
        enabled=True,
        source="test",
        fill_strategy=FillStrategy.DROP,
        period=50,
        field="close",
        operation=OperationType.MEAN,
    )


def test_feature_rolling_mean_drop(sin_wave_time_series, moving_average):
    res = moving_average.to_df(sin_wave_time_series, None)
    assert len(res) == 51
    assert res["rolling_mean_50"].isnull().sum() == 0


def test_feature_rolling_mean_backward_fill(sin_wave_time_series, moving_average):
    moving_average.fill_strategy = FillStrategy.BACKWARD_FILL
    res = moving_average.to_df(sin_wave_time_series, None)
    assert (res["rolling_mean_50"][0:48] == res["rolling_mean_50"][49]).all()
    assert len(res) == 100
    assert res["rolling_mean_50"].isnull().sum() == 0


def test_feature_rolling_mean_zero(sin_wave_time_series, moving_average):
    moving_average.fill_strategy = FillStrategy.ZERO
    res = moving_average.to_df(sin_wave_time_series, None)
    assert (res["rolling_mean_50"][0:48] == 0).all()
    assert len(res) == 100
    assert res["rolling_mean_50"].isnull().sum() == 0


def test_feature_rolling_mean_interpolate(sin_wave_time_series, moving_average):
    moving_average.fill_strategy = FillStrategy.INTERPOLATE
    res = moving_average.to_df(sin_wave_time_series, None)
    assert len(res) == 100
    assert res["rolling_mean_50"][0] == res["rolling_mean_50"][49]
    assert res["rolling_mean_50"][0] > res["rolling_mean_50"][50]
    assert res["rolling_mean_50"].isnull().sum() == 0


@pytest.fixture
def rsi():
    return RSI(
        type=FeatureType.MOVING_WINDOW,
        name="rsi",
        enabled=True,
        source="test",
        fill_strategy=FillStrategy.BACKWARD_FILL,
        period=14,
        field="close",
        ewm=False,
    )


def test_feature_rsi(sin_wave_time_series, rsi):
    res = rsi.to_df(sin_wave_time_series, None)
    assert res["rsi"].isnull().sum() == 0
    assert len(res) == 100


@pytest.fixture
def atr():
    return ATR(
        type=FeatureType.MOVING_WINDOW,
        name="atr",
        enabled=True,
        source="test",
        fill_strategy=FillStrategy.BACKWARD_FILL,
        period=14,
        field_high="high",
        field_low="low",
        field_close="close",
    )


def test_feature_atr(sin_wave_time_series, atr):
    res = atr.to_df(sin_wave_time_series, None)
    assert len(res) == 100
    assert res["atr"][0] == res["atr"][13]
    assert res["atr"].isnull().sum() == 0


@pytest.fixture
def bollinger_bands():
    return BollingerBands(
        type=FeatureType.BOLLINGER_BANDS,
        name="bollinger_bands",
        enabled=True,
        source="test",
        fill_strategy=FillStrategy.BACKWARD_FILL,
        period=20,
        field="close",
        ewm=False,
        alpha=2,
    )


def test_feature_bollinger_bands(sin_wave_time_series, bollinger_bands):
    res = bollinger_bands.to_df(sin_wave_time_series, None)
    assert len(res) == 100
    assert "bollinger_bands_upper" in res.columns
    assert "bollinger_bands_lower" in res.columns
    assert (res["bollinger_bands_upper"] >= res["bollinger_bands_lower"]).all()
    assert res["bollinger_bands_upper"].isnull().sum() == 0
    assert res["bollinger_bands_lower"].isnull().sum() == 0


@pytest.fixture
def macd():
    return MACD(
        type=FeatureType.MACD,
        name="macd",
        enabled=True,
        source="test",
        fill_strategy=FillStrategy.BACKWARD_FILL,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        field="close",
        ewm=False,
        signal_ewm=False,
    )


def test_feature_macd(sin_wave_time_series, macd):
    res = macd.to_df(sin_wave_time_series, None)
    assert len(res) == 100
    assert "macd" in res.columns
    assert "macd_signal" in res.columns
    assert "macd_hist" in res.columns
    assert (res["macd"] >= res["macd_signal"]).any() or (
        res["macd"] < res["macd_signal"]
    ).any()
    assert res["macd"].isnull().sum() == 0


@pytest.fixture
def mstd():
    return MSTD(
        type=FeatureType.MSTD,
        name="mstd",
        enabled=True,
        source="test",
        fill_strategy=FillStrategy.BACKWARD_FILL,
        field="close",
        ewm=False,
        window=20,
    )


def test_feature_mstd(sin_wave_time_series, mstd):
    res = mstd.to_df(sin_wave_time_series, None)
    assert "mstd" in res.columns
    assert res["mstd"].isnull().sum() == 0
    assert len(res) == 100


@pytest.fixture
def obv():
    return OBV(
        type=FeatureType.OBV,
        name="obv",
        enabled=True,
        source="test",
        fill_strategy=FillStrategy.BACKWARD_FILL,
        field_close="close",
        field_volume="volume",
    )


def test_feature_obv(sin_wave_time_series, obv):
    res = obv.to_df(sin_wave_time_series, None)
    assert len(res) == 100
    assert "obv" in res.columns
    assert res["obv"].isnull().sum() == 0


@pytest.fixture
def stochastic():
    return Stochastic(
        type=FeatureType.STOCHASTIC,
        name="stochastic",
        enabled=True,
        source="test",
        fill_strategy=FillStrategy.BACKWARD_FILL,
        period=14,
        field_high="high",
        field_low="low",
        field_close="close",
        k_window=14,
        d_window=3,
        ewm=False,
    )


def test_feature_stochastic(sin_wave_time_series, stochastic):
    res = stochastic.to_df(sin_wave_time_series, None)
    assert len(res) == 100
    assert "stochastic_k" in res.columns
    assert "stochastic_d" in res.columns
    assert res["stochastic_k"].isnull().sum() == 0
    assert res["stochastic_d"].isnull().sum() == 0


@pytest.fixture
def turbulence():
    return Turbulence(
        type=FeatureType.TURBULENCE,
        name="turbulence",
        enabled=True,
        source="test",
        fill_strategy=FillStrategy.BACKWARD_FILL,
        lookback=10,
        field="close",
    )


def test_feature_turbulence(sin_wave_time_series, turbulence):
    res = turbulence.to_df(sin_wave_time_series, None)
    assert len(res) == 100
    assert "turbulence" in res.columns
    assert res["turbulence"].isnull().sum() == 0
    assert (res["turbulence"] >= 0).all()  # Turbulence should be non-negative

    assert res["turbulence"].std() > 0, "Turbulence should vary for sine wave"
    assert (
        res["turbulence"].max() < 100
    ), "Turbulence shouldn't be extremely high for smooth sine wave"
