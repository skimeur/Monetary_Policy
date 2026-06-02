# Monetary Policy — Replications

Python replications of influential papers in **monetary economics**, by Eric Vansteenberghe.

> ⚠️ Independent replications for study purposes. Where modern or annual
> cross-country data are used in place of the original samples, results approximate
> the spirit of the original rather than reproducing its exact numbers.

Two replications are published here:

- **Sims (1980)**, *Macroeconomics and Reality* — [`code/sims1980_var.py`](code/sims1980_var.py)
- **Galí & Gertler (1999)**, *Inflation Dynamics: A Structural Econometric Analysis* — [`code/gali1999_replication.py`](code/gali1999_replication.py)

---

## Sims (1980) — VAR / SVAR

[`code/sims1980_var.py`](code/sims1980_var.py) replicates the empirical spirit of
Sims (1980) using annual JST Macrohistory data for `narrowm` (M1 proxy),
`rgdpbarro` (output), `unemp`, `wage`, `cpi`, and `imports` (import-price proxy).
Baseline recursive ordering: `narrowm, rgdpbarro, unemp, wage, cpi, imports`.

It implements:

- direct JST Release 6 download into a local `data/` cache,
- a reduced-form VAR with information-criterion lag selection,
- recursive (Cholesky) identification for orthogonalized impulse responses,
- forecast-error variance decomposition (FEVD),
- Granger / block-exogeneity tests (incl. money → unemployment, money → inflation),
- residual-whiteness and stability diagnostics,
- a robustness check across alternative recursive orderings.

```bash
python code/sims1980_var.py --country USA --maxlags 8 --ic aic --prefer dta --horizon 12
```

Use `--no-cache` to force a fresh JST download.

**Caveat.** A reduced-form VAR is a statistical forecasting system, not a structural
model on its own; recursive identification makes the variable ordering part of the
maintained hypothesis — hence the ordering-robustness check. This is an
approximation with annual data, not the exact frequency/sample of Sims (1980).

---

## Galí & Gertler (1999) — New Keynesian Phillips Curve

[`code/gali1999_replication.py`](code/gali1999_replication.py) estimates the new and
hybrid New Keynesian Phillips Curves by GMM, using the (log) labor income share as
the measure of real marginal cost. It covers:

- the reduced-form NKPC and the structural parameters (price stickiness θ, discount
  factor β) via two-step GMM with a Newey–West HAC weighting matrix;
- the hybrid model with rule-of-thumb price setters (forward/backward weights
  γ_f, γ_b and the degree of backwardness ω);
- both GMM normalizations, plus restricted-β and non-farm-deflator robustness variants;
- the original 1960Q1–1997Q4 sample and an extended sample to the latest data;
- a side-by-side comparison of the published coefficients with the replication.

```bash
python code/gali1999_replication.py --sample both
```

Use `--refresh` to re-download the FRED series.

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
- **FRED** — `GDPDEF`, `PRS85006173` (non-farm labor share), `IPDNBS`, `GS10`,
  `TB3MS`, `PPIACO` (Galí–Gertler replication), fetched via the public CSV endpoint.

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

**Eric Vansteenberghe.** If you use this code, please cite the original papers it replicates.
