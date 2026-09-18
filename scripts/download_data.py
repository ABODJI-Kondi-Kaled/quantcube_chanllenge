#!/usr/bin/env python3
"""Télécharge toutes les séries FRED et sauvegarde les snapshots dans data/raw/.

Nécessite FRED_API_KEY dans .env (copiez .env.example vers .env).
À lancer une seule fois ; les parquets générés sont committés dans le repo.
"""
from __future__ import annotations

from dotenv import load_dotenv

from quantcube_challenge.data.fred_client import fetch_series
from quantcube_challenge.data.registry import FRED_SERIES


def main() -> None:
    load_dotenv()

    print("Téléchargement des séries FRED...\n")
    errors: list[str] = []

    for code, spec in FRED_SERIES.items():
        print(f"  {code:<10} ({spec.frequency:<12}) ... ", end="", flush=True)
        try:
            series = fetch_series(spec, refresh=True)
            print(f"✓  {len(series)} observations")
        except Exception as exc:
            print(f"✗  {exc}")
            errors.append(code)

    print()
    if errors:
        print(f"Échec pour : {', '.join(errors)}")
    else:
        print("Toutes les séries sauvegardées dans data/raw/")


if __name__ == "__main__":
    main()
