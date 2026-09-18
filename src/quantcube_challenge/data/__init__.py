from __future__ import annotations

from quantcube_challenge.data.fred_client import fetch_series
from quantcube_challenge.data.registry import FRED_SERIES, SeriesSpec

__all__ = ["fetch_series", "FRED_SERIES", "SeriesSpec"]
