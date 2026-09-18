from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantcube_challenge.data.fred_client import _parse_observations, fetch_series
from quantcube_challenge.data.registry import FRED_SERIES


def _make_series(code: str = "PAYEMS") -> pd.Series:
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    series: pd.Series = pd.Series(
        [100.0, 101.0, 102.0], index=dates, dtype=float, name=code
    )
    series.index.name = "date"
    return series


def test_fetch_local_reads_parquet_without_network(tmp_path: Path) -> None:
    spec = FRED_SERIES["PAYEMS"]
    series = _make_series("PAYEMS")
    series.to_frame().to_parquet(tmp_path / "PAYEMS.parquet")

    with patch("quantcube_challenge.data.fred_client.requests.get") as mock_get:
        result = fetch_series(spec, refresh=False, data_dir=tmp_path)
        mock_get.assert_not_called()

    pd.testing.assert_series_equal(result, series)


def test_fetch_local_raises_if_parquet_missing(tmp_path: Path) -> None:
    spec = FRED_SERIES["PAYEMS"]
    with pytest.raises(FileNotFoundError, match="PAYEMS"):
        fetch_series(spec, refresh=False, data_dir=tmp_path)


def test_fetch_refresh_calls_api_and_saves_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test_key_123")
    spec = FRED_SERIES["GDPC1"]

    mock_response = MagicMock()
    obs = [
        {"date": "2020-01-01", "value": "21000.0"},
        {"date": "2020-04-01", "value": "."},
        {"date": "2020-07-01", "value": "21500.0"},
    ]
    mock_response.json.return_value = {"observations": obs}

    with patch(
        "quantcube_challenge.data.fred_client.requests.get",
        return_value=mock_response,
    ):
        result = fetch_series(spec, refresh=True, data_dir=tmp_path)

    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(21000.0)
    assert (tmp_path / "GDPC1.parquet").exists()


def test_parse_observations_filters_missing_values() -> None:
    observations = [
        {"date": "2020-01-01", "value": "100.0"},
        {"date": "2020-02-01", "value": "."},
        {"date": "2020-03-01", "value": "102.0"},
    ]
    result = _parse_observations(observations, "TEST")
    assert len(result) == 2
    assert result.index.name == "date"
    assert result.name == "TEST"
