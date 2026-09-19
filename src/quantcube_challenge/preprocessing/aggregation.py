from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class AggregationStrategy(Protocol):
    def aggregate(self, series: pd.Series) -> pd.Series: ...


class MeanAggregation:
    """Moyenne des observations disponibles dans le trimestre."""

    def aggregate(self, series: pd.Series) -> pd.Series:
        result: pd.Series = series.resample("QS").mean()
        return result


class LastValueAggregation:
    """Dernière observation disponible dans le trimestre."""

    def aggregate(self, series: pd.Series) -> pd.Series:
        result: pd.Series = series.resample("QS").last()
        return result


class WeightedAggregation:
    """Agrégation pondérée des 3 mois du trimestre (poids croissants par défaut).

    Pour les séries sub-mensuelles, agrège d'abord vers le mensuel via moyenne.
    Les poids doivent sommer à 1.
    """

    def __init__(self, weights: tuple[float, float, float] = (0.2, 0.3, 0.5)) -> None:
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Les poids doivent sommer à 1.")
        self._weights = np.array(weights, dtype=float)

    def aggregate(self, series: pd.Series) -> pd.Series:
        monthly = series.resample("MS").mean()
        quarters: dict[pd.Timestamp, float] = {}
        for q_start, group in monthly.groupby(pd.Grouper(freq="QS")):
            if len(group) == 3 and not group.isna().any():
                ts = pd.Timestamp(str(q_start))
                quarters[ts] = float(np.dot(self._weights, group.to_numpy(dtype=float)))
        return pd.Series(quarters, name=series.name)


AGGREGATION_MAP: dict[str, AggregationStrategy] = {
    "mean": MeanAggregation(),
    "last": LastValueAggregation(),
    "weighted": WeightedAggregation(),
}
