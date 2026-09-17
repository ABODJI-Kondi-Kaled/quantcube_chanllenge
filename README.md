# QuantCube Challenge — Nowcasting du PIB américain

Estimation du taux de croissance trimestriel annualisé du PIB réel américain (GDPC1)
avant sa publication officielle, à partir d'indicateurs FRED disponibles plus tôt.

## Installation

Prérequis : Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo>
cd quantcube_challenge
uv sync --all-extras
```

Pour utiliser votre propre clé API FRED (optionnel — uniquement pour rafraîchir les données) :

```bash
cp .env.example .env
# Éditez .env et renseignez FRED_API_KEY
```

## Reproduire l'analyse en 1 commande

```bash
make check          # ruff + mypy + pytest
```

Les données sont incluses dans le repo (`data/raw/`). Aucune clé API nécessaire.

Pour regénérer les données depuis FRED (nécessite `FRED_API_KEY` dans `.env`) :

```bash
python scripts/download_data.py
```

## Structure du projet

```
src/quantcube_challenge/
├── data/          # Ingestion FRED (F02)
├── preprocessing/ # Transformations Strategy (F03)
├── models/        # Modèles nowcast Template Method (F05-F07)
├── evaluation/    # Backtest expanding window (F08)
├── ragged_edge/   # Simulation disponibilité données (F09)
├── alternative/   # Google Trends bonus (F11)
└── reporting/     # Génération figures rapport (F12)

data/raw/          # Snapshot FRED versionné (parquet)
notebooks/         # Exploration et rapport interactif
reports/           # PDF final et figures
```

## Vérification qualité

```bash
make lint    # ruff (style + imports)
make test    # pytest
make check   # lint + mypy --strict + pytest
```
