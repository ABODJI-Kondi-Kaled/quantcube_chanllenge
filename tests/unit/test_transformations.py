from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcube_challenge.preprocessing.stationarity import stationarity_report
from quantcube_challenge.preprocessing.transformations import (
    TRANSFORMATION_MAP,
    AnnualizedGrowth,
    Diff,
    Identity,
    LogDiff,
    TransformationStrategy,
)


def _quarterly_series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="QS")
    return pd.Series(values, index=idx, dtype=float, name="TEST")


def _monthly_series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    return pd.Series(values, index=idx, dtype=float, name="TEST")


# --- AnnualizedGrowth ---


def test_annualized_growth_value() -> None:
    series = _quarterly_series([100.0, 102.0])
    result = AnnualizedGrowth().apply(series)
    expected = (1.02**4 - 1) * 100
    assert result.iloc[1] == pytest.approx(expected, rel=1e-5)


def test_annualized_growth_first_value_is_nan() -> None:
    series = _quarterly_series([100.0, 102.0])
    result = AnnualizedGrowth().apply(series)
    assert pd.isna(result.iloc[0])


# --- LogDiff ---


def test_log_diff_value() -> None:
    series = _monthly_series([100.0, 110.0])
    result = LogDiff().apply(series)
    assert result.iloc[1] == pytest.approx(np.log(110.0 / 100.0), rel=1e-5)


def test_log_diff_first_value_is_nan() -> None:
    series = _monthly_series([100.0, 110.0])
    result = LogDiff().apply(series)
    assert pd.isna(result.iloc[0])


# --- Diff ---


def test_diff_value() -> None:
    series = _monthly_series([10.0, 12.0, 11.0])
    result = Diff().apply(series)
    assert result.iloc[1] == pytest.approx(2.0)
    assert result.iloc[2] == pytest.approx(-1.0)


def test_diff_first_value_is_nan() -> None:
    series = _monthly_series([10.0, 12.0])
    result = Diff().apply(series)
    assert pd.isna(result.iloc[0])


# --- Identity ---


def test_identity_unchanged() -> None:
    series = _monthly_series([1.0, -1.0, 0.5])
    result = Identity().apply(series)
    pd.testing.assert_series_equal(result, series)


# --- Contrat commun : index préservé ---


def test_all_strategies_preserve_index() -> None:
    series = _monthly_series([100.0, 102.0, 104.0, 103.0, 105.0])
    strategies: list[TransformationStrategy] = [
        AnnualizedGrowth(),
        LogDiff(),
        Diff(),
        Identity(),
    ]
    for strategy in strategies:
        result = strategy.apply(series)
        assert result.index.equals(series.index), (
            f"{type(strategy).__name__} ne préserve pas l'index"
        )


# --- TRANSFORMATION_MAP ---


def test_transformation_map_covers_all_types() -> None:
    for key in ("annualized_growth", "log_diff", "diff", "none"):
        assert key in TRANSFORMATION_MAP


# --- Stationarité ---


def test_stationary_series_detected() -> None:
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 200), name="stationary")
    result = stationarity_report(series)
    assert result.is_stationary


def test_nonstationary_series_detected() -> None:
    rng = np.random.default_rng(42)
    series = pd.Series(np.cumsum(rng.normal(0, 1, 200)), name="random_walk")
    result = stationarity_report(series)
    assert not result.is_stationary


def test_stationarity_result_fields() -> None:
    series = pd.Series(np.random.default_rng(0).normal(0, 1, 100), name="s")
    result = stationarity_report(series)
    assert 0.0 <= result.adf_pvalue <= 1.0
    assert 0.0 <= result.kpss_pvalue <= 1.0
    assert result.series_name == "s"
