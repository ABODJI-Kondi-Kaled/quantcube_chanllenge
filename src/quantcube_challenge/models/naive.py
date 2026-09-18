from __future__ import annotations

import pandas as pd

from quantcube_challenge.models.base import BaseNowcastModel


class NaiveLastValue(BaseNowcastModel):
    """Benchmark : prédit la dernière valeur observée de y pour chaque période."""

    def _prepare_features(
        self,
        y: pd.Series,
        X: pd.DataFrame | None,
    ) -> tuple[pd.Series, pd.DataFrame | None]:
        return y.dropna(), None

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        pass  # Aucun paramètre à estimer

    def _predict(self, y: pd.Series, X: pd.DataFrame | None) -> pd.Series:
        # Prédit y_{t-1} pour chaque t : shift(1) puis forward-fill la dernière obs
        result: pd.Series = y.shift(1).ffill()
        return result
