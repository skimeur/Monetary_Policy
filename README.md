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
| **Galí, Gertler & López-Salido (2001)** — *European Inflation Dynamics* | New Keynesian Phillips Curve (GMM), euro area | [`code/gali1999_euro_replication.py`](code/gali1999_euro_replication.py) |

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
  normalizations; restricted-β and non-farm-deflator robustness variants; an
  **Anderson–Rubin weak-identification-robust confidence set** for the marginal-cost
  slope; a `--nlags` switch for instrument-count robustness; and a side-by-side
  comparison of the published coefficients with the replication.

```bash
python code/gali1999_replication.py --sample both
# --refresh re-downloads the FRED series; --nlags 2 uses fewer instruments
```

**Caveat.** With current-vintage data the labor-share–inflation link is weaker than
in the paper's 1998 vintage and the curve is weakly identified, so λ comes out
smaller, θ higher, and β is driven toward 1 — though the qualitative ranking
γ_f > γ_b (forward-looking pricing dominates) still holds. See *Weak identification*
below.

### Euro-area extension — Galí, Gertler & López-Salido (2001)

> **Reference.** Jordi Galí, Mark Gertler & J. David López-Salido (2001), "European
> inflation dynamics," *European Economic Review* **45**(7), 1237–1270 (erratum *EER*
> **47**(4), 759–760, 2003).

**Python replication →** [`code/gali1999_euro_replication.py`](code/gali1999_euro_replication.py)

GGL (2001) apply the same NKPC/GMM framework to the **euro area**. This standalone
script replicates it on modern data: inflation from the euro-area GDP deflator
(FRED), real marginal cost from the euro-area **labour income share** (Eurostat
national accounts, compensation of employees over GDP, adjusted for the
self-employed), instruments analogous to the US set, sample 1995Q1–latest. It
reproduces GGL's qualitative findings — a positive marginal-cost slope, high price
stickiness (θ ≈ 0.83), and an almost purely forward-looking hybrid (ω ≈ 0,
γ_f ≫ γ_b) — under the constant-returns (ξ = 1) normalization.

```bash
python code/gali1999_euro_replication.py
# --refresh re-downloads FRED + Eurostat series; --nlags sets instrument depth
```

### Weak identification & robust inference

GMM estimation of the New Keynesian Phillips Curve is **weakly identified**: the
moment conditions are nearly uninformative about the structural parameters, so
conventional GMM standard errors and the over-identification (*J*) test are
unreliable. This is now the central methodological issue for NKPC estimation. Two
diagnostics in this repository make it concrete:

- **Degenerate *J*-test.** With the paper's 25 instruments — a 25×25 HAC weight
  matrix estimated from only ~120–150 quarters — the *J*-test has almost no power and
  its *p*-values pin near 1.00. Shrinking the instrument set (`--nlags 2` or
  `--nlags 1`) restores power, and the model is still not rejected.
- **Anderson–Rubin robust confidence set.** The scripts compute the
  weak-instrument-robust **Anderson–Rubin / Stock–Wright *S*** confidence set for the
  marginal-cost slope λ (profiling out β on the continuously-updated objective). For
  **both** the US and the euro area, and for **every** instrument set tried (7–25
  instruments), this set is **unbounded** — once identification-robust inference is
  used the data are essentially uninformative about λ, and the tight conventional
  standard errors are illusory.

**How to do it properly.** Do not rely on Wald *t*-statistics / standard errors under
weak identification. Use **identification-robust inference**: the Anderson–Rubin /
Stock–Wright *S*-statistic and the Kleibergen *K*-statistic, evaluated on the
**continuously-updated (CUE)** GMM objective, and report confidence **sets** (which
may be unbounded or empty) rather than point estimates ± standard errors; keep the
instrument set parsimonious to limit many-instrument bias. See **Kleibergen &
Mavroeidis (2009)** and **Mavroeidis, Plagborg-Møller & Stock (2014)** in the
References.

---

## Repository structure

```
.
├── README.md
├── LICENSE
└── code/
    ├── sims1980_var.py            # Sims (1980) VAR / SVAR (CLI)
    ├── gali1999_replication.py    # Galí & Gertler (1999) NKPC (GMM, US)
    └── gali1999_euro_replication.py  # Galí–Gertler–López-Salido (2001) NKPC (GMM, euro area)
```

`data/` (downloaded datasets and caches) is created on first run and is git-ignored.

---

## Data sources

- **JST Macrohistory Database, Release 6** — auto-downloaded from
  [macrohistory.net](https://www.macrohistory.net/) (Sims replication).
- **FRED** (Federal Reserve Bank of St. Louis) — US series `GDPDEF`, `PRS85006173`
  (non-farm labor share), `IPDNBS`, `GS10`, `TB3MS`, `COMPNFB`, `PPIACO`, `GDPC1`,
  and euro-area aggregates `CLVMNACSCAB1GQEA19`, `CPMNACSCAB1GQEA19`,
  `IRLTLT01EZM156N`, `IR3TIB01EZQ156N`.
- **Eurostat** — euro-area national accounts (`namq_10_gdp`, `namq_10_a10_e`) for the
  labour income share (Galí–Gertler–López-Salido replication).

---

## Setup

Python 3.10+.

```bash
pip install numpy pandas scipy statsmodels requests matplotlib
```

> **Working-directory note.** The `gali1999_*` scripts set a local working directory
> via `os.chdir(...)` near the top; edit it to your own checkout (or remove it and run
> from the repo root) before executing.

---

## References

- **Sims, Christopher A.** (1980). "Macroeconomics and Reality." *Econometrica*
  48(1), 1–48. DOI: [10.2307/1912017](https://doi.org/10.2307/1912017)
- **Galí, Jordi & Mark Gertler** (1999). "Inflation dynamics: A structural
  econometric analysis." *Journal of Monetary Economics* 44(2), 195–222.
  DOI: [10.1016/S0304-3932(99)00023-9](https://doi.org/10.1016/S0304-3932(99)00023-9)
- **Galí, Jordi, Mark Gertler & J. David López-Salido** (2001). "European inflation
  dynamics." *European Economic Review* 45(7), 1237–1270.
  [RePEc](https://ideas.repec.org/a/eee/eecrev/v45y2001i7p1237-1270.html) ·
  NBER w8218 DOI: [10.3386/w8218](https://doi.org/10.3386/w8218). Erratum: *EER*
  47(4), 759–760 (2003).
- **Kleibergen, Frank & Sophocles Mavroeidis** (2009). "Weak instrument robust tests
  in GMM and the new Keynesian Phillips curve." *Journal of Business & Economic
  Statistics* 27(3), 293–311.
  DOI: [10.1198/jbes.2009.08280](https://doi.org/10.1198/jbes.2009.08280)
- **Mavroeidis, Sophocles, Mikkel Plagborg-Møller & James H. Stock** (2014).
  "Empirical evidence on inflation expectations in the New Keynesian Phillips curve."
  *Journal of Economic Literature* 52(1), 124–188.
  DOI: [10.1257/jel.52.1.124](https://doi.org/10.1257/jel.52.1.124)

---

## Author

**Eric Vansteenberghe.** If you use this code, please cite the original papers listed
in the References.

---

<sub>Keywords: monetary policy · macroeconometrics · replication · Python · VAR ·
SVAR · structural VAR · impulse response · FEVD · New Keynesian Phillips curve ·
NKPC · hybrid Phillips curve · GMM · two-step GMM · continuously-updated GMM · CUE ·
Newey–West HAC · weak identification · Anderson–Rubin · Stock–Wright S-statistic ·
identification-robust inference · Kleibergen–Mavroeidis · Calvo pricing · inflation
dynamics · marginal cost · labor share · euro area · Galí–Gertler ·
Galí–Gertler–López-Salido · FRED · Eurostat · JST Macrohistory.</sub>
