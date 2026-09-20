from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    common = actual.index.intersection(predicted.dropna().index)
    diff = actual.loc[common] - predicted.loc[common]
    return float(np.sqrt((diff**2).mean()))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    common = actual.index.intersection(predicted.dropna().index)
    return float((actual.loc[common] - predicted.loc[common]).abs().mean())


def diebold_mariano(
    actual: pd.Series,
    pred1: pd.Series,
    pred2: pd.Series,
    h: int = 1,
) -> tuple[float, float]:
    """Test de Diebold-Mariano (1995) — H0 : les deux modèles ont la même précision.

    Retourne (statistique DM, p-value).
    Implémentation simplifiée avec correction de variance Newey-West (h lags).
    """
    common = actual.index.intersection(pred1.dropna().index).intersection(
        pred2.dropna().index
    )
    e1 = (actual.loc[common] - pred1.loc[common]).to_numpy(dtype=float)
    e2 = (actual.loc[common] - pred2.loc[common]).to_numpy(dtype=float)
    d = e1**2 - e2**2

    n = len(d)
    d_mean = float(d.mean())
    gamma0 = float(((d - d_mean) ** 2).mean())
    gamma_lags = sum(
        float(((d[lag:] - d_mean) * (d[:-lag] - d_mean)).mean())
        for lag in range(1, h + 1)
    )
    var_d = (gamma0 + 2 * gamma_lags) / n

    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = d_mean / math.sqrt(var_d)
    p_value = 2.0 * (1.0 - _norm_cdf(abs(dm_stat)))
    return dm_stat, p_value


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
