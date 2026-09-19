from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from quantcube_challenge.models.bridge import BridgeEquation
from quantcube_challenge.preprocessing.aggregation import AggregationStrategy

_ALPHAS: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
_L1_RATIOS: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]


class RidgeBridge(BridgeEquation):
    """BridgeEquation avec pénalité Ridge (L2).

    α optimal par validation croisée temporelle (TimeSeriesSplit).
    """

    def __init__(
        self,
        aggregation: AggregationStrategy | None = None,
        alphas: list[float] | None = None,
        n_splits: int = 5,
    ) -> None:
        super().__init__(aggregation)
        self._alphas: list[float] = alphas if alphas is not None else _ALPHAS
        self._n_splits = n_splits
        self._best_alpha: float | None = None

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        assert X is not None
        self._feature_names = list(X.columns)
        tscv = TimeSeriesSplit(n_splits=self._n_splits)
        gs = GridSearchCV(
            Ridge(),
            param_grid={"alpha": self._alphas},
            cv=tscv,
            scoring="neg_mean_squared_error",
        )
        gs.fit(X.to_numpy(), y.to_numpy())
        self._best_alpha = float(gs.best_params_["alpha"])
        self._model = gs.best_estimator_


class ElasticNetBridge(BridgeEquation):
    """BridgeEquation avec pénalité ElasticNet (L1 + L2).

    Recommandé quand les prédicteurs sont corrélés par groupes.
    α et l1_ratio optimaux par validation croisée temporelle.
    """

    def __init__(
        self,
        aggregation: AggregationStrategy | None = None,
        alphas: list[float] | None = None,
        l1_ratios: list[float] | None = None,
        n_splits: int = 5,
    ) -> None:
        super().__init__(aggregation)
        self._alphas = alphas if alphas is not None else _ALPHAS
        self._l1_ratios: list[float] = (
            l1_ratios if l1_ratios is not None else _L1_RATIOS
        )
        self._n_splits = n_splits
        self._best_alpha: float | None = None
        self._best_l1_ratio: float | None = None

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        assert X is not None
        self._feature_names = list(X.columns)
        tscv = TimeSeriesSplit(n_splits=self._n_splits)
        gs = GridSearchCV(
            ElasticNet(max_iter=10_000),
            param_grid={"alpha": self._alphas, "l1_ratio": self._l1_ratios},
            cv=tscv,
            scoring="neg_mean_squared_error",
        )
        gs.fit(X.to_numpy(), y.to_numpy())
        self._best_alpha = float(gs.best_params_["alpha"])
        self._best_l1_ratio = float(gs.best_params_["l1_ratio"])
        self._model = gs.best_estimator_


class PCABridge(BridgeEquation):
    """BridgeEquation avec réduction PCA avant OLS.

    Les données sont standardisées avant PCA.
    La première composante s'interprète comme le cycle des affaires.
    """

    def __init__(
        self,
        aggregation: AggregationStrategy | None = None,
        n_components: int = 2,
    ) -> None:
        super().__init__(aggregation)
        self._n_components = n_components
        self._scaler: StandardScaler = StandardScaler()
        self._pca: PCA = PCA(n_components=n_components)
        self._original_feature_names: list[str] = []

    def _prepare_features(
        self,
        y: pd.Series,
        X: pd.DataFrame | None,
    ) -> tuple[pd.Series, pd.DataFrame | None]:
        y_clean, X_q = super()._prepare_features(y, X)
        assert X_q is not None

        self._original_feature_names = list(X_q.columns)
        X_scaled = self._scaler.fit_transform(X_q.to_numpy())
        X_pca = self._pca.fit_transform(X_scaled)

        col_names = [f"PC{i + 1}" for i in range(self._n_components)]
        X_pca_df = pd.DataFrame(X_pca, index=y_clean.index, columns=col_names)
        return y_clean, X_pca_df

    @property
    def explained_variance_ratio_(self) -> list[float]:
        return [float(v) for v in self._pca.explained_variance_ratio_]

    @property
    def loadings_(self) -> pd.DataFrame:
        """Contribution de chaque indicateur original à chaque composante."""
        col_names = [f"PC{i + 1}" for i in range(self._n_components)]
        return pd.DataFrame(
            self._pca.components_.T,
            index=self._original_feature_names,
            columns=col_names,
        )
