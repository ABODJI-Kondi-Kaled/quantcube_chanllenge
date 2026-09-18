from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from quantcube_challenge.models.base import BaseNowcastModel


class AR1(BaseNowcastModel):
    """Modèle AR(1) : y_t = α + β·y_{t-1} + ε_t, estimé par OLS."""

    def __init__(self) -> None:
        self._model: LinearRegression = LinearRegression()

    def _prepare_features(
        self,
        y: pd.Series,
        X: pd.DataFrame | None,
    ) -> tuple[pd.Series, pd.DataFrame | None]:
        y_clean = y.dropna()
        lag = y_clean.shift(1).dropna()
        common = y_clean.index.intersection(lag.index)
        X_lag = pd.DataFrame({"y_lag1": lag.loc[common]})
        return y_clean.loc[common], X_lag

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        assert X is not None, "AR1 attend une matrice de features (lag)"
        self._model.fit(X.to_numpy(), y.to_numpy())

    def _predict(self, y: pd.Series, X: pd.DataFrame | None) -> pd.Series:
        assert X is not None
        preds: np.ndarray[tuple[int], np.dtype[np.float64]] = self._model.predict(
            X.to_numpy()
        )
        return pd.Series(preds, index=y.index, name=y.name)
