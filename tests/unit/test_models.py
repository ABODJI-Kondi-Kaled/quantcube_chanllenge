from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcube_challenge.models import AR1, BridgeEquation, NaiveLastValue


def _toy_series(n: int = 20) -> pd.Series:
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(2.0, 1.0, n), index=idx, name="gdp")


class TestNaiveLastValue:
    def test_predicts_lagged_value(self) -> None:
        s = _toy_series()
        preds = NaiveLastValue().fit_predict(s)
        # La prédiction en t doit être la valeur en t-1
        assert preds.iloc[1] == pytest.approx(s.iloc[0])
        assert preds.iloc[5] == pytest.approx(s.iloc[4])

    def test_output_index_matches_input(self) -> None:
        s = _toy_series()
        preds = NaiveLastValue().fit_predict(s)
        assert preds.index.equals(s.index)

    def test_first_prediction_is_nan(self) -> None:
        s = _toy_series()
        preds = NaiveLastValue().fit_predict(s)
        assert np.isnan(preds.iloc[0])

    def test_no_x_required(self) -> None:
        s = _toy_series()
        preds = NaiveLastValue().fit_predict(s, X=None)
        assert len(preds) == len(s)


class TestAR1:
    def test_output_length(self) -> None:
        s = _toy_series()
        preds = AR1().fit_predict(s)
        assert len(preds) == len(s) - 1  # 1 obs perdue au lag

    def test_output_index_aligned(self) -> None:
        s = _toy_series()
        preds = AR1().fit_predict(s)
        # L'index doit être un sous-ensemble contigu de s.index
        assert preds.index[0] == s.index[1]

    def test_predictions_are_finite(self) -> None:
        s = _toy_series()
        preds = AR1().fit_predict(s)
        assert np.all(np.isfinite(preds.to_numpy()))

    def test_ar1_on_random_walk(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2000-01-01", periods=50, freq="QS")
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 50)), index=idx, name="rw")
        preds = AR1().fit_predict(rw)
        assert len(preds) == 49

    def test_constant_series_predicts_constant(self) -> None:
        idx = pd.date_range("2000-01-01", periods=10, freq="QS")
        s = pd.Series([3.0] * 10, index=idx, name="const")
        preds = AR1().fit_predict(s)
        assert np.allclose(preds.to_numpy(), 3.0, atol=1e-6)


def _toy_bridge_data(
    n_quarters: int = 40,
) -> tuple[pd.Series, pd.DataFrame]:
    """Série trimestrielle y et indicateurs mensuels X pour tester BridgeEquation."""
    q_idx = pd.date_range("2000-01-01", periods=n_quarters, freq="QS")
    m_idx = pd.date_range("2000-01-01", periods=n_quarters * 3, freq="MS")
    rng = np.random.default_rng(7)
    y = pd.Series(rng.normal(2.0, 1.0, n_quarters), index=q_idx, name="gdp")
    X = pd.DataFrame(
        {
            "indpro": rng.normal(0.003, 0.01, n_quarters * 3),
            "payems": rng.normal(0.002, 0.008, n_quarters * 3),
        },
        index=m_idx,
    )
    return y, X


class TestBridgeEquation:
    def test_output_is_series(self) -> None:
        y, X = _toy_bridge_data()
        preds = BridgeEquation().fit_predict(y, X)
        assert isinstance(preds, pd.Series)

    def test_output_index_quarterly(self) -> None:
        y, X = _toy_bridge_data()
        preds = BridgeEquation().fit_predict(y, X)
        assert len(preds) > 0

    def test_predictions_are_finite(self) -> None:
        y, X = _toy_bridge_data()
        preds = BridgeEquation().fit_predict(y, X)
        assert np.all(np.isfinite(preds.to_numpy()))

    def test_coef_keys_match_columns(self) -> None:
        y, X = _toy_bridge_data()
        bridge = BridgeEquation()
        bridge.fit_predict(y, X)
        assert set(bridge.coef_.keys()) == {"indpro", "payems"}

    def test_requires_x(self) -> None:
        y, _ = _toy_bridge_data()
        with pytest.raises(AssertionError):
            BridgeEquation().fit_predict(y, X=None)

    def test_length_equals_aligned_quarters(self) -> None:
        y, X = _toy_bridge_data(n_quarters=20)
        preds = BridgeEquation().fit_predict(y, X)
        assert len(preds) == 20
