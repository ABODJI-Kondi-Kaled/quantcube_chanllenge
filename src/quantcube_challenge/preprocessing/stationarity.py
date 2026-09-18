from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass(frozen=True)
class StationarityResult:
    series_name: str
    adf_pvalue: float
    kpss_pvalue: float
    adf_stationary: bool  # ADF rejette H0 "racine unitaire" → p < 0.05
    kpss_stationary: bool  # KPSS ne rejette pas H0 "stationnaire" → p > 0.05
    is_stationary: bool  # True si les deux tests concordent


def stationarity_report(series: pd.Series) -> StationarityResult:
    """Teste la stationnarité via ADF et KPSS croisés.

    ADF  : H0 = racine unitaire (non stationnaire). Rejeter H0 → stationnaire.
    KPSS : H0 = stationnaire.   Ne pas rejeter H0  → stationnaire.
    Conclusion fiable uniquement quand les deux tests concordent.
    """
    clean: pd.Series = series.dropna()
    name = str(series.name) if series.name is not None else "unnamed"

    adf_stat, adf_pvalue, *_ = adfuller(clean, autolag="AIC", result_object=False)
    kpss_stat, kpss_pvalue, *_ = kpss(
        clean, regression="c", nlags="auto", result_object=False
    )

    adf_stationary = float(adf_pvalue) < 0.05
    kpss_stationary = float(kpss_pvalue) > 0.05

    return StationarityResult(
        series_name=name,
        adf_pvalue=float(adf_pvalue),
        kpss_pvalue=float(kpss_pvalue),
        adf_stationary=adf_stationary,
        kpss_stationary=kpss_stationary,
        is_stationary=adf_stationary and kpss_stationary,
    )
