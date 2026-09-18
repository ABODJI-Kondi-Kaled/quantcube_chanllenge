from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Frequency = Literal["monthly", "quarterly", "weekly"]
TransformationType = Literal["annualized_growth", "log_diff", "diff", "none"]


@dataclass(frozen=True)
class SeriesSpec:
    code: str
    description: str
    frequency: Frequency
    publication_lag_days: int  # délai réel entre fin de période et publication
    seasonally_adjusted: bool
    units: str
    target_transformation: TransformationType


FRED_SERIES: dict[str, SeriesSpec] = {
    "GDPC1": SeriesSpec(
        code="GDPC1",
        description="PIB réel américain (cible)",
        frequency="quarterly",
        publication_lag_days=30,
        seasonally_adjusted=True,
        units="Billions of Chained 2017 Dollars",
        target_transformation="annualized_growth",
    ),
    "CFNAI": SeriesSpec(
        code="CFNAI",
        description="Chicago Fed National Activity Index",
        frequency="monthly",
        publication_lag_days=28,
        seasonally_adjusted=True,
        units="Index",
        target_transformation="none",  # déjà stationnaire (z-score de 85 indicateurs)
    ),
    "INDPRO": SeriesSpec(
        code="INDPRO",
        description="Production industrielle",
        frequency="monthly",
        publication_lag_days=15,
        seasonally_adjusted=True,
        units="Index 2017=100",
        target_transformation="log_diff",
    ),
    "PAYEMS": SeriesSpec(
        code="PAYEMS",
        description="Emploi salarié non agricole",
        frequency="monthly",
        publication_lag_days=7,
        seasonally_adjusted=True,
        units="Thousands of Persons",
        target_transformation="log_diff",
    ),
    "ICSA": SeriesSpec(
        code="ICSA",
        description="Inscriptions hebdomadaires au chômage",
        frequency="weekly",
        publication_lag_days=5,
        seasonally_adjusted=True,
        units="Number",
        target_transformation="log_diff",
    ),
    "RSAFS": SeriesSpec(
        code="RSAFS",
        description="Ventes au détail",
        frequency="monthly",
        publication_lag_days=15,
        seasonally_adjusted=True,
        units="Millions of Dollars",
        target_transformation="log_diff",
    ),
    "UMCSENT": SeriesSpec(
        code="UMCSENT",
        description="Indice de confiance des consommateurs Michigan",
        frequency="monthly",
        publication_lag_days=15,
        seasonally_adjusted=False,
        units="Index 1966:Q1=100",
        target_transformation="diff",
    ),
    "NFCI": SeriesSpec(
        code="NFCI",
        description="National Financial Conditions Index",
        frequency="weekly",
        publication_lag_days=7,
        seasonally_adjusted=True,
        units="Index",
        target_transformation="none",  # déjà stationnaire (z-score centré sur 0)
    ),
    "PERMIT": SeriesSpec(
        code="PERMIT",
        description="Permis de construire",
        frequency="monthly",
        publication_lag_days=15,
        seasonally_adjusted=True,
        units="Thousands of Units",
        target_transformation="log_diff",
    ),
}
