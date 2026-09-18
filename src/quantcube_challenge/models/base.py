from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseNowcastModel(ABC):
    """Template Method : fit_predict() orchestre _prepare_features, _fit, _predict."""

    def fit_predict(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Entraîne le modèle et retourne les prédictions in-sample."""
        y_clean, X_clean = self._prepare_features(y, X)
        self._fit(y_clean, X_clean)
        return self._predict(y_clean, X_clean)

    @abstractmethod
    def _prepare_features(
        self,
        y: pd.Series,
        X: pd.DataFrame | None,
    ) -> tuple[pd.Series, pd.DataFrame | None]:
        """Nettoie / aligne y et X. Chaque sous-classe définit ses besoins."""
        ...

    @abstractmethod
    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        """Estime les paramètres sur y (et X si nécessaire)."""
        ...

    @abstractmethod
    def _predict(self, y: pd.Series, X: pd.DataFrame | None) -> pd.Series:
        """Retourne les valeurs prédites alignées sur l'index de y."""
        ...
