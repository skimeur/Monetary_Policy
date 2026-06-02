# Monetary Policy — Paper Replications in Python

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Topic: Monetary Economics](https://img.shields.io/badge/topic-monetary%20economics-brightgreen)
![Type: Paper Replication](https://img.shields.io/badge/type-paper%20replication-orange)

Reproducible **Python replications** of landmark papers in **monetary economics**.
Each replication reproduces the original paper end to end — its **theory** (the model
and estimating equations), its **data** (downloaded from public sources), and its
**empirical results** — in a single runnable script. By Eric Vansteenberghe.

> ⚠️ Independent replications for study purposes. Where modern or annual
> cross-country data are used in place of the original samples, results approximate
> the spirit of the original rather than reproducing its exact numbers.

**Replications in this repository**

| Original paper | Method | Python replication |
|----------------|--------|--------------------|
| **Sims (1980)** — *Macroeconomics and Reality* | VAR / SVAR | [`code/sims1980_var.py`](code/sims1980_var.py) |
| **Galí & Gertler (1999)** — *Inflation Dynamics: A Structural Econometric Analysis* | New Keynesian Phillips Curve (GMM) | [`code/gali1999_replication.py`](code/gali1999_replication.py) |

---

## Sims (1980) — Macroeconomics and Reality

> **Original paper.** Christopher A. Sims (1980), "Macroeconomics and Reality,"
> *Econometrica* **48**(1), 1–48. DOI: [10.2307/1912017](https://doi.org/10.2307/1912017)

**Python replication →** [`code/sims1980_var.py`](code/sims1980_var.py)

A self-contained replication of Sims's reduced-form VAR and recursive structural VAR
(SVAR) analysis of money, output, and prices.

- **📐 Theory** — reduced-form vector autoregression; recursive (Cholesky)
  identification of structural shocks; orthogonalized impulse–response functions;
  forecast-error variance decomposition (FEVD); Granger / block-exogeneity tests
  (e.g. money → unemployment, money → inflation); residual-whiteness and stability
  diagnostics.
- **📊 Data** — annual **JST Macrohistory Database** (Release 6), auto-downloaded:
  `narrowm` (M1 proxy), `rgdpbarro` (output), `unemp`, `wage`, `cpi`, `imports`
  (import-price proxy). Baseline recursive ordering
  `narrowm, rgdpbarro, unemp, wage, cpi, imports`.
- **🐍 Code** — one command-line script with information-criterion lag selection and
  a robustness check across alternative recursive orderings.

```bash
python code/sims1980_var.py --country USA --maxlags 8 --ic aic --prefer dta --horizon 12
# --no-cache forces a fresh JST download
```

**Caveat.** A reduced-form VAR is a statistical forecasting system, not a structural
model on its own; recursive identification makes the variable ordering part of the
maintained hypothesis — hence the ordering-robustness check. This is an
approximation with annual data, not the exact frequency/sample of Sims (1980).

---

## Galí & Gertler (1999) — Inflation Dynamics

> **Original paper.** Jordi Galí and Mark Gertler (1999), "Inflation dynamics: A
> structural econometric analysis," *Journal of Monetary Economics* **44**(2),
> 195–222. DOI: [10.1016/S0304-3932(99)00023-9](https://doi.org/10.1016/S0304-3932(99)00023-9)

**Python replication →** [`code/gali1999_replication.py`](code/gali1999_replication.py)

A full replication of the estimation of the New Keynesian Phillips Curve (NKPC) with
real marginal cost measured by the labor income share.

- **📐 Theory** — the forward-looking NKPC and the **hybrid** NKPC with rule-of-thumb
  price setters; structural parameters (Calvo price stickiness θ, discount factor β,
  backward-looking share ω) recovered from the rational-expectations orthogonality
  conditions; reduced-form vs. structural coefficients (λ, γ_f, γ_b).
- **📊 Data** — quarterly **FRED** series: GDP deflator (`GDPDEF`) for inflation, the
  non-farm-business labor share (`PRS85006173`) for marginal cost, plus instruments
  (`IPDNBS`, `GS10`, `TB3MS`, `COMPNFB`, `PPIACO`, `GDPC1`). Original 1960Q1–1997Q4
  sample and an extended sample to the latest data.
- **🐍 Code** — two-step GMM with a Newey–West HAC weighting matrix; both moment
  normalizations; restricted-β and non-farm-deflator robustness variants; and a
  side-by-side comparison of the published coefficients with the replication.

```bash
python code/gali1999_replication.py --sample both
# --refresh re-downloads the FRED series
```

**Caveat.** With current-vintage data the labor-share–inflation link is weaker than
in the paper's 1998 vintage and the curve is weakly identified, so λ comes out
smaller, θ higher, and β is driven toward 1 — though the qualitative ranking
γ_f > γ_b (forward-looking pricing dominates) still holds.

---

## Repository structure

```
.
├── README.md
├── LICENSE
└── code/
    ├── sims1980_var.py          # Sims (1980) VAR / SVAR (CLI)
    └── gali1999_replication.py  # Galí & Gertler (1999) NKPC (GMM)
```

`data/` (downloaded datasets and caches) is created on first run and is git-ignored.

---

## Data sources

- **JST Macrohistory Database, Release 6** — auto-downloaded from
  [macrohistory.net](https://www.macrohistory.net/) (Sims replication).
- **FRED** (Federal Reserve Bank of St. Louis) — `GDPDEF`, `PRS85006173` (non-farm
  labor share), `IPDNBS`, `GS10`, `TB3MS`, `COMPNFB`, `PPIACO`, `GDPC1`
  (Galí–Gertler replication), fetched via the public CSV endpoint.

---

## Setup

Python 3.10+.

```bash
pip install numpy pandas scipy statsmodels requests matplotlib
```

> **Working-directory note.** `gali1999_replication.py` sets a local working
> directory via `os.chdir(...)` near the top; edit it to your own checkout (or
> remove it and run from the repo root) before executing.

---

## Author

**Eric Vansteenberghe.** If you use this code, please cite the original papers
linked above.

---

<sub>Keywords: monetary policy · macroeconometrics · replication · Python · VAR ·
SVAR · structural VAR · impulse response · FEVD · New Keynesian Phillips curve ·
NKPC · GMM · Calvo pricing · inflation dynamics · marginal cost · FRED · JST
Macrohistory.</sub>
