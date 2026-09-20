from __future__ import annotations

import numpy as np
import pandas as pd

from quantcube_challenge.evaluation import ExpandingWindow, ragged_x_test
from quantcube_challenge.evaluation.backtest import _quarter_month_cutoff
from quantcube_challenge.models import BridgeEquation
from quantcube_challenge.preprocessing.aggregation import MeanAggregation


def _toy_data(n: int = 50) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(42)
    q_idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    m_idx = pd.date_range("2000-01-01", periods=n * 3, freq="MS")
    y = pd.Series(rng.normal(2.0, 1.0, n), index=q_idx, name="gdp")
    X = pd.DataFrame(
        {"x1": rng.normal(0, 1, n * 3), "x2": rng.normal(0, 1, n * 3)},
        index=m_idx,
    )
    return y, X


class TestQuarterMonthCutoff:
    def test_m1_end_of_first_month(self) -> None:
        assert _quarter_month_cutoff(pd.Timestamp("2026-07-01"), 1) == pd.Timestamp(
            "2026-07-31"
        )

    def test_m2_end_of_second_month(self) -> None:
        assert _quarter_month_cutoff(pd.Timestamp("2026-07-01"), 2) == pd.Timestamp(
            "2026-08-31"
        )

    def test_m3_end_of_quarter(self) -> None:
        assert _quarter_month_cutoff(pd.Timestamp("2026-07-01"), 3) == pd.Timestamp(
            "2026-09-30"
        )

    def test_q4_no_year_overflow(self) -> None:
        assert _quarter_month_cutoff(pd.Timestamp("2025-10-01"), 3) == pd.Timestamp(
            "2025-12-31"
        )

    def test_q1_m3_march(self) -> None:
        assert _quarter_month_cutoff(pd.Timestamp("2026-01-01"), 3) == pd.Timestamp(
            "2026-03-31"
        )


class TestRaggedXTest:
    def test_returns_one_row(self) -> None:
        _, X = _toy_data()
        q = pd.Timestamp("2001-04-01")
        result = ragged_x_test(X, q, 1, MeanAggregation())
        assert result is not None
        assert len(result) == 1

    def test_m3_matches_full_quarterly_aggregation(self) -> None:
        _, X = _toy_data()
        q = pd.Timestamp("2001-04-01")
        r_m3 = ragged_x_test(X, q, 3, MeanAggregation())
        X_q = pd.DataFrame(
            {col: MeanAggregation().aggregate(X[col]) for col in X.columns}
        )
        assert r_m3 is not None
        pd.testing.assert_frame_equal(r_m3, X_q.loc[[q]], check_freq=False)

    def test_none_when_no_data(self) -> None:
        _, X = _toy_data()
        future = pd.Timestamp("2030-01-01")
        result = ragged_x_test(X, future, 1, MeanAggregation())
        assert result is None


class TestRaggedBacktest:
    def test_m3_same_as_default(self) -> None:
        y, X = _toy_data()
        r_default = ExpandingWindow(min_train=20).run(BridgeEquation(), y, X)
        r_m3 = ExpandingWindow(min_train=20, month_in_quarter=3).run(
            BridgeEquation(), y, X
        )
        pd.testing.assert_frame_equal(r_default.data, r_m3.data)

    def test_m1_not_empty(self) -> None:
        y, X = _toy_data()
        result = ExpandingWindow(min_train=20, month_in_quarter=1).run(
            BridgeEquation(), y, X
        )
        assert len(result) > 0

    def test_all_months_produce_predictions(self) -> None:
        y, X = _toy_data()
        for m in [1, 2, 3]:
            r = ExpandingWindow(min_train=20, month_in_quarter=m).run(
                BridgeEquation(), y, X
            )
            assert len(r) > 0, f"M{m} ne produit aucune prédiction"

    def test_no_x_unaffected_by_month(self) -> None:
        from quantcube_challenge.models import AR1

        y, _ = _toy_data()
        r1 = ExpandingWindow(min_train=20, month_in_quarter=1).run(AR1(), y)
        r3 = ExpandingWindow(min_train=20, month_in_quarter=3).run(AR1(), y)
        pd.testing.assert_frame_equal(r1.data, r3.data)
