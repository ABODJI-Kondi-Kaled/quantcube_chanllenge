from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcube_challenge.preprocessing.aggregation import (
    LastValueAggregation,
    MeanAggregation,
    WeightedAggregation,
)


def _monthly(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=idx, name="x")


def _weekly(values: list[float], start: str = "2020-01-06") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="W-MON")
    return pd.Series(values, index=idx, name="x")


class TestMeanAggregation:
    def test_quarterly_mean_monthly(self) -> None:
        s = _monthly([1.0, 2.0, 3.0])  # Q1-2020 : jan=1, fev=2, mar=3
        result = MeanAggregation().aggregate(s)
        assert result.loc["2020-01-01"] == pytest.approx(2.0)

    def test_two_quarters(self) -> None:
        s = _monthly([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = MeanAggregation().aggregate(s)
        assert result.loc["2020-01-01"] == pytest.approx(2.0)
        assert result.loc["2020-04-01"] == pytest.approx(5.0)

    def test_weekly_mean(self) -> None:
        # ~4 semaines en janvier → moyenne ~ valeur constante
        s = _weekly([10.0] * 13)  # Q1-2020 ≈ 13 semaines
        result = MeanAggregation().aggregate(s)
        assert not result.dropna().empty

    def test_output_is_series(self) -> None:
        s = _monthly([1.0, 2.0, 3.0])
        result = MeanAggregation().aggregate(s)
        assert isinstance(result, pd.Series)


class TestLastValueAggregation:
    def test_takes_last_month(self) -> None:
        s = _monthly([1.0, 2.0, 3.0])  # Q1 : last = mars = 3.0
        result = LastValueAggregation().aggregate(s)
        assert result.loc["2020-01-01"] == pytest.approx(3.0)

    def test_two_quarters(self) -> None:
        s = _monthly([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = LastValueAggregation().aggregate(s)
        assert result.loc["2020-01-01"] == pytest.approx(3.0)
        assert result.loc["2020-04-01"] == pytest.approx(6.0)


class TestWeightedAggregation:
    def test_weighted_sum(self) -> None:
        s = _monthly([1.0, 2.0, 3.0])
        # 0.2*1 + 0.3*2 + 0.5*3 = 0.2 + 0.6 + 1.5 = 2.3
        result = WeightedAggregation(weights=(0.2, 0.3, 0.5)).aggregate(s)
        assert result.loc["2020-01-01"] == pytest.approx(2.3)

    def test_equal_weights_equals_mean(self) -> None:
        s = _monthly([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        w = WeightedAggregation(weights=(1 / 3, 1 / 3, 1 / 3)).aggregate(s)
        m = MeanAggregation().aggregate(s)
        common = w.dropna().index.intersection(m.dropna().index)
        np.testing.assert_allclose(
            w.loc[common].to_numpy(), m.loc[common].to_numpy(), atol=1e-10
        )

    def test_invalid_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="sommer"):
            WeightedAggregation(weights=(0.1, 0.2, 0.3))
