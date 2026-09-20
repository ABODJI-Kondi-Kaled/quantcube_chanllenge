# QuantCube Challenge — Nowcasting du PIB américain

Estimation du taux de croissance trimestriel annualisé du PIB réel américain
(GDPC1) **avant sa publication officielle**, à partir d'indicateurs FRED
disponibles à plus haute fréquence.

**Auteur :** Kondi Kaled ABODJI — `kondi.kaled.abodji@gmail.com`
**Repository :** <https://github.com/ABODJI-Kondi-Kaled/quantcube_chanllenge>

---

## Que lire en premier

1. **`reports/rapport_nowcasting.pdf`** — note de synthèse 3 pages (livrable principal du challenge).
2. **`notebooks/00_synthese.ipynb`** — notebook unique qui reproduit toute
   l'analyse (chargement des données → backtest → nowcast Q3-2026) en
   17 cellules. À ouvrir en second pour vérifier les chiffres du rapport.
3. **`notebooks/01_*` à `08_*`** — 8 notebooks détaillés qui documentent le
   raisonnement pas à pas (exploration, choix de la cible, baseline, bridge
   equation, régularisation, backtest, ragged edge, robustesse Covid).
   À consulter pour aller plus loin sur un point précis.

---

## Installation (2 minutes)

**Prérequis :** Python 3.11+, [`uv`](https://docs.astral.sh/uv/) (`pip install uv`
ou `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/ABODJI-Kondi-Kaled/quantcube_chanllenge.git
cd quantcube_chanllenge
uv sync --all-extras
```

Les données FRED sont **versionnées** dans `data/raw/` (fichiers parquet),
**aucune clé API n'est nécessaire** pour reproduire l'analyse.

## Reproduire l'analyse

```bash
make check     # ruff + mypy --strict + pytest (79 tests, ~5 s)
```

Pour ré-exécuter le notebook de synthèse et regénérer les figures :

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/00_synthese.ipynb --inplace
```

Pour regénérer le PDF du rapport (nécessite `pandoc` + `pdflatex`) :

```bash
pandoc reports/rapport_nowcasting.md -o reports/rapport_nowcasting.pdf --pdf-engine=pdflatex
```

## Rafraîchir les données depuis FRED (optionnel)

Nécessite une clé API FRED gratuite (<https://fred.stlouisfed.org/docs/api/api_key.html>).

```bash
cp .env.example .env
# Éditez .env et renseignez FRED_API_KEY=xxxxx
uv run python scripts/download_data.py
```

---

## Structure du projet

```
src/quantcube_challenge/
├── data/            # Client FRED + snapshot parquet
├── preprocessing/   # Transformations (Log-diff, Diff...) + agrégations (Strategy)
├── models/          # NaiveLastValue, AR(1), BridgeEquation, Ridge, ElasticNet, PCA (Template Method)
└── evaluation/      # ExpandingWindow, RollingWindow, ragged_x_test, Diebold-Mariano

notebooks/
├── 00_synthese.ipynb            # Lecture rapide — reproduit tout
├── 01_data_exploration.ipynb    # Séries FRED, stationnarité
├── 02_target_choice.ipynb       # Niveau vs différencié
├── 03_baseline_models.ipynb     # Naive + AR(1)
├── 04_bridge_equation.ipynb     # OLS Bridge + stratégies d'agrégation
├── 05_regularization.ipynb      # Ridge / ElasticNet / PCA
├── 06_backtest.ipynb            # Expanding window (évaluation honnête)
├── 07_ragged_edge.ipynb         # Nowcast M1 / M2 / M3
└── 08_covid_robustness.ipynb    # RMSE avec vs sans 2020

tests/unit/                      # 79 tests (pytest)
data/raw/                        # 9 séries FRED en parquet (versionnées)
reports/                         # PDF + figures PNG
scripts/download_data.py         # Rafraîchit data/raw/ depuis FRED
```

## Choix méthodologiques (résumé)

- **Cible :** taux de croissance annualisé — convention BEA/Fed, stationnaire, lisible.
- **Modèle recommandé :** **ElasticNet Bridge** (RMSE hors-Covid = 1,84 pp,
  gain +16 % vs AR(1)).
- **Protocole d'évaluation :** backtest en fenêtre extensible, comparaison
  sur fenêtre commune 1992–2026 (137 trimestres), avec/sans Covid.
- **Architecture :** patterns *Strategy* (agrégations, transformations) et
  *Template Method* (modèles), typage `mypy --strict`, 79 tests unitaires.

Détails complets dans `reports/rapport_nowcasting.pdf`.

## Qualité

```bash
make lint      # ruff (style + imports)
make test      # pytest
make check     # lint + mypy --strict + pytest
```
