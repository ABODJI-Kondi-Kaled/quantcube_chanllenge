from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcube_challenge.evaluation import (
    BacktestResult,
    ExpandingWindow,
    RollingWindow,
    diebold_mariano,
    mae,
    rmse,
)
from quantcube_challenge.models import AR1, BridgeEquation, NaiveLastValue


def _toy_data(n: int = 50) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(99)
    q_idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    m_idx = pd.date_range("2000-01-01", periods=n * 3, freq="MS")
    y = pd.Series(rng.normal(2.0, 1.0, n), index=q_idx, name="gdp")
    X = pd.DataFrame(
        {"x1": rng.normal(0, 1, n * 3), "x2": rng.normal(0, 1, n * 3)},
        index=m_idx,
    )
    return y, X


class TestBacktestResult:
    def test_rmse_constant_error(self) -> None:
        data = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "predicted": [1.5, 2.5, 3.5]})
        assert BacktestResult(data).rmse == pytest.approx(0.5)

    def test_mae_constant_error(self) -> None:
        data = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "predicted": [1.5, 2.5, 3.5]})
        assert BacktestResult(data).mae == pytest.approx(0.5)

    def test_len(self) -> None:
        data = pd.DataFrame({"actual": [1.0, 2.0], "predicted": [1.0, 2.0]})
        assert len(BacktestResult(data)) == 2


class TestExpandingWindow:
    def test_naive_result_length(self) -> None:
        y, _ = _toy_data()
        result = ExpandingWindow(min_train=20).run(NaiveLastValue(), y)
        assert len(result) == 50 - 20 - 1  # 29 prédictions

    def test_naive_no_lookahead(self) -> None:
        y, _ = _toy_data()
        result = ExpandingWindow(min_train=20).run(NaiveLastValue(), y)
        # pred[t] doit être y[t-1] : le 21e trimestre est prédit = y[20e]
        pred_val = float(result.data["predicted"].iloc[0])
        actual_prev = float(y.iloc[20])  # min_train=20 → 21e obs prédite = y[20]
        assert pred_val == pytest.approx(actual_prev)

    def test_ar1_result_not_empty(self) -> None:
        y, _ = _toy_data()
        result = ExpandingWindow(min_train=20).run(AR1(), y)
        assert len(result) > 0

    def test_bridge_result_not_empty(self) -> None:
        y, X = _toy_data()
        result = ExpandingWindow(min_train=20).run(BridgeEquation(), y, X)
        assert len(result) > 0

    def test_rmse_positive(self) -> None:
        y, _ = _toy_data()
        result = ExpandingWindow(min_train=20).run(AR1(), y)
        assert result.rmse > 0


class TestRollingWindow:
    def test_result_length(self) -> None:
        y, _ = _toy_data()
        result = RollingWindow(window=20).run(NaiveLastValue(), y)
        assert len(result) == 50 - 20  # 30 prédictions

    def test_naive_identical_rmse_to_expanding(self) -> None:
        y, _ = _toy_data()
        exp = ExpandingWindow(min_train=20).run(NaiveLastValue(), y)
        roll = RollingWindow(window=20).run(NaiveLastValue(), y)
        # NaiveLastValue ignore la taille de fenêtre → même RMSE sur le chevauchement
        common = exp.data.index.intersection(roll.data.index)
        np.testing.assert_allclose(
            exp.data.loc[common, "predicted"].to_numpy(),
            roll.data.loc[common, "predicted"].to_numpy(),
        )


class TestMetrics:
    def test_rmse_perfect(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        assert rmse(s, s) == pytest.approx(0.0)

    def test_mae_perfect(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        assert mae(s, s) == pytest.approx(0.0)

    def test_diebold_mariano_returns_tuple(self) -> None:
        rng = np.random.default_rng(0)
        y = pd.Series(rng.normal(0, 1, 50))
        p1 = y + pd.Series(rng.normal(0, 0.5, 50))
        p2 = y + pd.Series(rng.normal(0, 1.0, 50))
        stat, pval = diebold_mariano(y, p1, p2)
        assert isinstance(stat, float)
        assert 0.0 <= pval <= 1.0

    def test_diebold_mariano_perfect_model_wins(self) -> None:
        rng = np.random.default_rng(1)
        y = pd.Series(rng.normal(0, 1, 100))
        perfect = y.copy()
        bad = y + pd.Series(rng.normal(0, 2.0, 100))
        stat, _ = diebold_mariano(y, perfect, bad)
        assert stat < 0  # modèle 1 meilleur → stat négative
