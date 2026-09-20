from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantcube_challenge.evaluation.metrics import mae, rmse
from quantcube_challenge.models.base import BaseNowcastModel
from quantcube_challenge.preprocessing.aggregation import (
    AggregationStrategy,
    MeanAggregation,
)


@dataclass(frozen=True)
class BacktestResult:
    """Résultat d'un backtest : prédictions et réalisations alignées."""

    data: pd.DataFrame  # colonnes : "actual", "predicted" — index DatetimeIndex

    @property
    def rmse(self) -> float:
        return rmse(self.data["actual"], self.data["predicted"])

    @property
    def mae(self) -> float:
        return mae(self.data["actual"], self.data["predicted"])

    def __len__(self) -> int:
        return len(self.data)


def _aggregate_x(X: pd.DataFrame, aggregation: AggregationStrategy) -> pd.DataFrame:
    return pd.DataFrame({col: aggregation.aggregate(X[col]) for col in X.columns})


def _quarter_month_cutoff(
    quarter_start: pd.Timestamp, month_in_quarter: int
) -> pd.Timestamp:
    """Retourne la fin du n-ième mois d'un trimestre donné."""
    m = quarter_start.month + month_in_quarter - 1
    y = quarter_start.year + (m - 1) // 12
    m = ((m - 1) % 12) + 1
    return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)


def ragged_x_test(
    X: pd.DataFrame,
    quarter_start: pd.Timestamp,
    month_in_quarter: int,
    aggregation: AggregationStrategy,
) -> pd.DataFrame | None:
    """Agrège X pour un trimestre partiel (ragged edge).

    Retourne un DataFrame à index trimestriel avec la moyenne des
    `month_in_quarter` premiers mois du trimestre `quarter_start`.
    Retourne None si aucune donnée n'est disponible dans la fenêtre.
    """
    cutoff = _quarter_month_cutoff(quarter_start, month_in_quarter)
    mask = (X.index >= quarter_start) & (X.index <= cutoff)
    x_raw = X.loc[mask]
    if x_raw.empty:
        return None
    x_agg = pd.DataFrame(
        {col: aggregation.aggregate(x_raw[col]) for col in x_raw.columns}
    )
    x_q = x_agg.loc[x_agg.index == quarter_start]
    return x_q if not x_q.empty else None


class ExpandingWindow:
    """Backtest en fenêtre extensible (expanding window).

    À chaque pas t, entraîne sur [début, t] et prédit t+1.
    `month_in_quarter` simule le ragged edge : 1 = seulement le premier mois
    du trimestre disponible, 2 = deux mois, 3 = trimestre complet (défaut).
    """

    def __init__(
        self,
        min_train: int = 20,
        aggregation: AggregationStrategy | None = None,
        month_in_quarter: int = 3,
    ) -> None:
        self._min_train = min_train
        self._aggregation: AggregationStrategy = (
            aggregation if aggregation is not None else MeanAggregation()
        )
        self._month_in_quarter = month_in_quarter

    def run(
        self,
        model: BaseNowcastModel,
        y: pd.Series,
        X: pd.DataFrame | None = None,
    ) -> BacktestResult:
        y_q = y.dropna()
        X_q = _aggregate_x(X, self._aggregation) if X is not None else None

        records: list[dict[str, object]] = []

        for i in range(self._min_train, len(y_q) - 1):
            train_end = y_q.index[i]
            test_date = y_q.index[i + 1]

            y_train = y_q.iloc[: i + 1]
            x_train = (
                X_q.loc[X_q.index <= train_end].dropna() if X_q is not None else None
            )

            if self._month_in_quarter == 3 or X is None:
                x_test: pd.DataFrame | None = (
                    X_q.loc[[test_date]]
                    if X_q is not None and test_date in X_q.index
                    else None
                )
            else:
                x_test = ragged_x_test(
                    X, test_date, self._month_in_quarter, self._aggregation
                )

            if x_train is not None and x_train.empty:
                continue
            if X_q is not None and (x_test is None or x_test.dropna().empty):
                continue

            model.fit(y_train, x_train)
            preds = model.predict(y_train, x_test)

            if len(preds) == 0:
                continue

            records.append(
                {
                    "date": test_date,
                    "actual": float(y_q.loc[test_date]),
                    "predicted": float(preds.iloc[0]),
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return BacktestResult(
                pd.DataFrame(columns=["actual", "predicted"], dtype=float)
            )
        df = df.set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        return BacktestResult(df.astype(float))


class RollingWindow:
    """Backtest en fenêtre roulante (taille fixe).

    N'utilise que les `window` derniers trimestres pour entraîner.
    Utile pour détecter des changements de régime récents.
    """

    def __init__(
        self,
        window: int = 40,
        aggregation: AggregationStrategy | None = None,
    ) -> None:
        self._window = window
        self._aggregation: AggregationStrategy = (
            aggregation if aggregation is not None else MeanAggregation()
        )

    def run(
        self,
        model: BaseNowcastModel,
        y: pd.Series,
        X: pd.DataFrame | None = None,
    ) -> BacktestResult:
        y_q = y.dropna()
        X_q = _aggregate_x(X, self._aggregation) if X is not None else None

        records: list[dict[str, object]] = []

        for i in range(self._window - 1, len(y_q) - 1):
            train_start = y_q.index[i - self._window + 1]
            train_end = y_q.index[i]
            test_date = y_q.index[i + 1]

            y_train = y_q.loc[train_start:train_end]
            x_train = (
                X_q.loc[(X_q.index >= train_start) & (X_q.index <= train_end)].dropna()
                if X_q is not None
                else None
            )
            x_test = (
                X_q.loc[[test_date]]
                if X_q is not None and test_date in X_q.index
                else None
            )

            if x_train is not None and x_train.empty:
                continue
            if X_q is not None and (x_test is None or x_test.dropna().empty):
                continue

            model.fit(y_train, x_train)
            preds = model.predict(y_train, x_test)

            if len(preds) == 0:
                continue

            records.append(
                {
                    "date": test_date,
                    "actual": float(y_q.loc[test_date]),
                    "predicted": float(preds.iloc[0]),
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return BacktestResult(
                pd.DataFrame(columns=["actual", "predicted"], dtype=float)
            )
        df = df.set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        return BacktestResult(df.astype(float))
