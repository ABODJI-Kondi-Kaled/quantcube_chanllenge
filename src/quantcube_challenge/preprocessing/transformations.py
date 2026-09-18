from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from quantcube_challenge.data.registry import TransformationType


class TransformationStrategy(Protocol):
    def apply(self, series: pd.Series) -> pd.Series: ...


class AnnualizedGrowth:
    """Taux de croissance trimestriel annualisé : ((x_t / x_{t-1})^4 - 1) * 100."""

    def apply(self, series: pd.Series) -> pd.Series:
        result: pd.Series = (series / series.shift(1)) ** 4 - 1
        return result * 100


class LogDiff:
    """Log-différence : log(x_t) - log(x_{t-1})."""

    def apply(self, series: pd.Series) -> pd.Series:
        log_values = pd.Series(
            np.log(series.to_numpy(dtype=float)),
            index=series.index,
            name=series.name,
        )
        result: pd.Series = log_values.diff()
        return result


class Diff:
    """Différence simple : x_t - x_{t-1}."""

    def apply(self, series: pd.Series) -> pd.Series:
        result: pd.Series = series.diff()
        return result


class Identity:
    """Aucune transformation — série déjà stationnaire."""

    def apply(self, series: pd.Series) -> pd.Series:
        return series.copy()


TRANSFORMATION_MAP: dict[TransformationType, TransformationStrategy] = {
    "annualized_growth": AnnualizedGrowth(),
    "log_diff": LogDiff(),
    "diff": Diff(),
    "none": Identity(),
}


def to_annualized_growth(gdp: pd.Series) -> pd.Series:
    """Convertit GDPC1 en taux de croissance trimestriel annualisé (%).

    Fonction standalone réutilisée dans tous les notebooks et modèles.
    Équivalent à AnnualizedGrowth().apply(gdp).dropna().
    """
    return AnnualizedGrowth().apply(gdp).dropna()
