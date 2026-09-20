---
title: "Nowcasting du PIB américain — Note de synthèse"
author: "Kaled Abodji"
date: "Septembre 2026"
geometry: "margin=2cm"
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{caption}
  - \captionsetup{font=small}
  - \usepackage[utf8]{inputenc}
---

## 1. Question et approche de modélisation

**Objectif :** estimer le taux de croissance du PIB réel américain (GDPC1) pour le trimestre courant, avant sa publication officielle par le BEA, en exploitant des indicateurs mensuels et hebdomadaires disponibles plus tôt.

L'approche retenue est l'**équation de pont** (*bridge equation*) : des indicateurs à fréquence mensuelle ou hebdomadaire sont agrégés à la fréquence trimestrielle, puis utilisés comme régresseurs dans un modèle de régression pour prédire le PIB. Trois variantes ont été comparées — OLS, Ridge et ElasticNet — complétées par une réduction dimensionnelle (PCA + OLS) et deux références naïves (dernière valeur observée, AR(1)).

## 2. Choix de la variable cible : taux annualisé

La cible est le **taux de croissance trimestriel annualisé**, défini comme :

$$g_t = \left[\left(\frac{\text{GDP}_t}{\text{GDP}_{t-1}}\right)^4 - 1\right] \times 100$$

**Justification :** (i) c'est la convention standard du BEA et de la Fed — tous les indicateurs FRED de référence utilisent cette unité, ce qui facilite la lecture et la comparaison ; (ii) la série en niveau présente une tendance stochastique qui induit une régression fallacieuse ; (iii) la première différence simple (taux trimestriel non annualisé) est valide mais moins lisible. Le taux annualisé est donc le meilleur compromis entre stationnarité et interprétabilité économique.

## 3. Données utilisées

**8 indicateurs FRED officiels**, couvrant les quatre dimensions principales du cycle :

| Indicateur | Catégorie | Transformation | Fréquence |
|:-----------|:----------|:---------------|:----------|
| CFNAI | Activité globale | Niveau | Mensuelle |
| INDPRO | Production industrielle | Log-diff | Mensuelle |
| PAYEMS | Emploi non-agricole | Log-diff | Mensuelle |
| ICSA | Allocations chômage | Log-diff | Hebdomadaire |
| RSAFS | Ventes au détail | Log-diff | Mensuelle |
| UMCSENT | Confiance consommateur | Différence | Mensuelle |
| NFCI | Conditions financières | Niveau | Hebdomadaire |
| PERMIT | Permis de construire | Log-diff | Mensuelle |

Les indicateurs hebdomadaires (ICSA, NFCI) et mensuels sont agrégés vers le trimestriel par moyenne arithmétique avant estimation (*mean aggregation*). RSAFS, disponible depuis 1992 seulement, délimite la fenêtre d'évaluation effective à 1992–2026.

## 4. Résultats principaux

**Protocole d'évaluation :** backtest en fenêtre extensible (*expanding window*) — à chaque trimestre $t$, le modèle est entraîné strictement sur $[t_0, t]$ et prédit $t+1$, sans accès aux données futures. Ce protocole élimine tout *look-ahead bias*. Toutes les métriques sont calculées sur la **fenêtre commune** (137 trimestres, 1992–2026) pour une comparaison équitable.

\begin{table}[H]
\centering
\caption{RMSE (pp annualisés) — backtest expanding window, fenêtre commune 1992–2026}
\begin{tabular}{lrrrrr}
\toprule
Modèle & RMSE total & RMSE hors-Covid & Impact Covid & Gain total & Gain hors-Covid \\
\midrule
NaiveLastValue & 6,81 & 2,62 & 4,19 & --34,8\% & --19,9\% \\
AR(1) & 5,05 & 2,18 & 2,87 & réf. & réf. \\
OLS Bridge & 3,46 & 2,29 & 1,18 & +31,5\% & --4,8\% \\
Ridge & 3,26 & 1,85 & 1,41 & +35,6\% & +15,3\% \\
\textbf{ElasticNet} & \textbf{3,21} & \textbf{1,84} & 1,38 & \textbf{+36,5\%} & \textbf{+15,9\%} \\
PCA (2 comp.) & 3,27 & 1,69 & 1,57 & +35,4\% & +22,4\% \\
\bottomrule
\end{tabular}
\end{table}

**Lecture clé :** l'OLS Bridge, malgré son gain apparent (+31,5 %), sous-performe AR(1) hors-Covid (--4,8 %). Ce résultat, contre-intuitif, s'explique par la multicollinéarité entre PAYEMS et INDPRO (coefficients OLS instables : PAYEMS = +755, INDPRO = --25). La régularisation (Ridge, ElasticNet) résout ce problème et maintient un gain structurel de +15–22 % même hors de la période de crise.

![RMSE selon l'avancement dans le trimestre (ElasticNet, fenêtre commune)](/home/vastolordess/Documents/others/personal_info/new/pfe_cv/quantcube/quantcube_challenge/reports/figures/07_ragged_edge_rmse.png){ width=85% }

**Analyse ragged edge :** le graphique montre l'évolution du RMSE selon le nombre de mois disponibles. Dès M1 (un seul mois), ElasticNet bat AR(1) de +12 %. La précision est maximale en M2 (+46 %), car le deuxième mois capture l'essentiel de la dynamique trimestrielle ; le troisième mois apporte peu de signal supplémentaire net (bruit de fin de trimestre).

**Nowcast Q3-2026 :** entraîné sur l'intégralité des données historiques, ElasticNet prédit **+3,3 pp annualisé** pour Q3-2026 (scénarios M1/M2/M3 convergents), contre +3,0 pp pour AR(1). Ce résultat suggère une croissance américaine modérée mais positive au troisième trimestre 2026.

## 5. Hypothèses, limites et perspectives

**Limites principales :**

- *Biais de révision (data vintage)* : le backtest utilise les données FRED dans leur version la plus récente (révisée). En conditions réelles, les praticiens disposent de données préliminaires, moins fiables. La base ALFRED (Archival FRED) permettrait d'éliminer ce biais, au prix d'une complexité supplémentaire significative.

- *Faible puissance statistique :* avec 137 observations communes, le test de Diebold-Mariano entre Bridge et AR(1) n'est pas significatif (p = 0,32). La différence de RMSE est économiquement significative mais statistiquement non confirmée — ce qui est habituel dans cette littérature avec des séries trimestrielles courtes.

- *Covid comme outlier exogène :* Q2-2020 (--28 pp) est imprévisible par tout modèle économétrique. Il gonfle uniformément le RMSE de tous les modèles (écart de 1,2 à 4,2 pp selon le modèle) sans changer le classement.

**Perspective non implémentée — Google Trends :** l'intégration de signaux alternatifs (volume de recherches pour "unemployment", "recession", "layoffs" via l'API *pytrends*) aurait pu enrichir le modèle. Les requêtes Google sont publiées en temps réel, sans délai ni révision, ce qui constitue un avantage théorique sur les indicateurs FRED. Cependant, trois obstacles ont motivé l'abandon de cette piste : (i) l'API Google Trends est non officielle et sujette à des limitations de débit imprévisibles ; (ii) les données ne couvrent que 2004+, réduisant encore la fenêtre d'évaluation ; (iii) la sélection *a posteriori* des mots-clés introduit un *look-ahead bias* dans les données elles-mêmes. Cette piste reste une extension naturelle pour un système de production.

## 6. Usage des outils d'IA

Claude Code (Anthropic, modèle Sonnet 4.6) a été utilisé pour l'assistance à la correction de certain bugs dans le backtest et à la vérification du coverage destest unitaires (on aurait pu utiliser sonarqube à la place mais plus de temps de configuration). Toutes les décisions de modélisation — choix des indicateurs, justification du taux annualisé, interprétation des résultats, identification du look-ahead bias dans les notebooks exploratoires — ont été prises et vérifiées manuellement.Les résultats numériques ont été interprétés indépendamment des suggestions de l'outil.
