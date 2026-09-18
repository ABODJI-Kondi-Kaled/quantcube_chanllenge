from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from quantcube_challenge.data.registry import SeriesSpec

_FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_PROJECT_ROOT = Path(__file__).parents[3]
_DATA_RAW = _PROJECT_ROOT / "data" / "raw"


def fetch_series(
    spec: SeriesSpec,
    refresh: bool = False,
    data_dir: Path | None = None,
) -> pd.Series:
    """Retourne la série brute FRED pour `spec`.

    Par défaut (refresh=False) : lecture du snapshot parquet local.
    Avec refresh=True : appel à l'API FRED officielle puis mise à jour du parquet.
    """
    raw_dir = data_dir if data_dir is not None else _DATA_RAW
    parquet_path = raw_dir / f"{spec.code}.parquet"

    if not refresh:
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Snapshot local absent pour {spec.code}. "
                "Lancez : python scripts/download_data.py"
            )
        return pd.read_parquet(parquet_path)[spec.code]

    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise OSError(
            "FRED_API_KEY absent. Copiez .env.example vers .env et renseignez la clé."
        )

    response = requests.get(
        _FRED_BASE_URL,
        params={
            "series_id": spec.code,
            "api_key": api_key,
            "file_type": "json",
        },
        timeout=30,
    )
    response.raise_for_status()

    observations: list[dict[str, str]] = response.json()["observations"]
    series = _parse_observations(observations, spec.code)

    raw_dir.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_parquet(parquet_path)

    return series


def _parse_observations(observations: list[dict[str, str]], code: str) -> pd.Series:
    dates = pd.to_datetime([obs["date"] for obs in observations if obs["value"] != "."])
    values = [float(obs["value"]) for obs in observations if obs["value"] != "."]
    series: pd.Series = pd.Series(values, index=dates, dtype=float, name=code)
    series.index.name = "date"
    return series
