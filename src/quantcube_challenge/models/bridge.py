from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

from quantcube_challenge.models.base import BaseNowcastModel
from quantcube_challenge.preprocessing.aggregation import (
    AggregationStrategy,
    MeanAggregation,
)


class BridgeEquation(BaseNowcastModel):
    """Équation de pont : agrège X vers le trimestriel puis OLS.

    y_q = α + β₁·agg(X₁_q) + β₂·agg(X₂_q) + … + ε
    """

    def __init__(self, aggregation: AggregationStrategy | None = None) -> None:
        self._aggregation: AggregationStrategy = (
            aggregation if aggregation is not None else MeanAggregation()
        )
        self._model: LinearRegression = LinearRegression()
        self._feature_names: list[str] = []

    def _prepare_features(
        self,
        y: pd.Series,
        X: pd.DataFrame | None,
    ) -> tuple[pd.Series, pd.DataFrame | None]:
        assert X is not None, "BridgeEquation requiert une matrice X d'indicateurs."

        X_q = pd.DataFrame(
            {col: self._aggregation.aggregate(X[col]) for col in X.columns}
        )

        y_clean = y.dropna()
        X_clean = X_q.dropna()
        common = y_clean.index.intersection(X_clean.index)

        return y_clean.loc[common], X_clean.loc[common]

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        assert X is not None
        self._feature_names = list(X.columns)
        self._model.fit(X.to_numpy(), y.to_numpy())

    def _predict(self, y: pd.Series, X: pd.DataFrame | None) -> pd.Series:
        assert X is not None
        preds = self._model.predict(X.to_numpy())
        return pd.Series(preds, index=y.index, name=y.name)

    @property
    def coef_(self) -> dict[str, float]:
        """Coefficients OLS estimés par nom d'indicateur."""
        return {k: float(v) for k, v in zip(self._feature_names, self._model.coef_)}

    @property
    def intercept_(self) -> float:
        return float(self._model.intercept_)
