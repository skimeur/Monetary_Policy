# Monetary Policy — Replications

Replications and worked derivations of foundational and modern papers in
**monetary economics**, by Eric Vansteenberghe.

Each paper is either re-derived analytically (full theorem-by-theorem proofs in
the accompanying LaTeX document) or replicated empirically in Python, and often
both. The write-up lives in [`pdf/vansteenberghe_MP.tex`](pdf/vansteenberghe_MP.tex)
(compiled to [`pdf/vansteenberghe_MP.pdf`](pdf/vansteenberghe_MP.pdf)); the code
lives in [`code/`](code/).

> ⚠️ These are **independent replications for study purposes**. Where modern data
> or annual cross-country data are used in place of the original samples, results
> approximate the spirit of the original rather than reproducing its exact numbers.

---

## Papers covered

The numbering follows the sections of the paper. "Analytical" means the result is
derived/proved in the LaTeX document; "Empirical" means there is a script in
[`code/`](code/).

| # | Paper | Type | Code |
|---|-------|------|------|
| 1 | Akerlof (1970), *The Market for "Lemons"* | Analytical | — |
| 2 | Stiglitz & Weiss (1981), *Credit Rationing in Markets with Imperfect Information* | Analytical (Thms 1–12) | — |
| 3 | **Sims (1980), *Macroeconomics and Reality*** | Analytical + Empirical | [`sims1980_var.py`](code/sims1980_var.py), [`sims_1980_replication.py`](code/sims_1980_replication.py), [`sims_1980_replication_GMD.py`](code/sims_1980_replication_GMD.py) |
| 4 | Taylor (1993), *Discretion versus Policy Rules in Practice* | Empirical | [`taylor_1993_replication.py`](code/taylor_1993_replication.py) |
| 5 | Kydland & Prescott (1977), *Rules Rather than Discretion* | Analytical + Empirical | [`kydland1977_replication.py`](code/kydland1977_replication.py) |
| 6 | Lucas (1976), *Econometric Policy Evaluation: A Critique* | Analytical | — |
| 7 | Muth (1961), *Rational Expectations and the Theory of Price Movements* | Analytical | — |
| 8 | Christiano, Eichenbaum & Evans (2005), *Nominal Rigidities…* | Analytical | — |
| 9 | Sims (2003), *Implications of Rational Inattention* | Analytical + Empirical | [`sim2003_replication.py`](code/sim2003_replication.py) |
| 10 | Leeper (1991), *Equilibria under 'Active' and 'Passive' Monetary and Fiscal Policies* | Analytical | — |
| 11 | Sargent & Wallace (1981), *Some Unpleasant Monetarist Arithmetic* | Analytical | — |
| 12 | **Galí & Gertler (1999), *Inflation Dynamics: A Structural Econometric Analysis*** | Analytical + Empirical | [`gali1999_replication.py`](code/gali1999_replication.py) |
| 13 | **Stock & Watson (2007), *Why Has U.S. Inflation Become Harder to Forecast?*** | Analytical + Empirical | [`stock2007_*.py`](code/) (see below) |
| 14 | Jarociński & Karadi (2020), *Deconstructing Monetary Policy Surprises* | Analytical | — |
| 15 | Ajello, Favara, Marchal & Szőke (2024), *Financial Conditions and Risks to the Economic Outlook* | Empirical | [`ajello2024.py`](code/ajello2024.py) |

### Supporting empirical scripts

| Script | What it does |
|--------|--------------|
| [`M1_GDP_cointegration_test.py`](code/M1_GDP_cointegration_test.py) | ADF / Johansen cointegration of money base, core CPI and real GDP (FRED) — supports the money–prices–income discussion in §3.7. |
| [`QTM_test_replication.py`](code/QTM_test_replication.py) | Long-run Quantity Theory of Money check on JST cross-country data (narrow/broad money, CPI, real GDP). |
| [`euribor_inflation_DFR.py`](code/euribor_inflation_DFR.py) | Fetches and plots EURIBOR tenors, the ECB Deposit Facility Rate and HICP inflation from the ECB Data Portal API. |
| [`bodacc_procedures_count.py`](code/bodacc_procedures_count.py), [`bodacc_procedures_count_date.py`](code/bodacc_procedures_count_date.py) | Pull and aggregate French insolvency proceedings from the BODACC open-data API (collective procedures over time). |

### Stock & Watson (2007) script map

| Script | Object replicated |
|--------|-------------------|
| [`stock2007_1_2.py`](code/stock2007_1_2.py) | §1.2 pseudo out-of-sample direct multi-step forecasts: AR(AIC), Atkeson–Ohanian, backward-looking Phillips curve. |
| [`stock2007_2_1.py`](code/stock2007_2_1.py) | Table 2: std. dev. and autocorrelations of Δπ, plus 90% CI for the largest AR root (Stock 1991 ADF-t inversion). |
| [`stock2007_2_1_modern.py`](code/stock2007_2_1_modern.py), [`…_data_check.py`](code/stock2007_2_1_modern_data_check.py) | Same as above but sourcing `GDPDEF` from FRED instead of the archival `.q47` files. |
| [`stock2007_2_2.py`](code/stock2007_2_2.py) | Table 3: IMA(1,1) / unobserved-components model and nested-model tests. |
| [`stock2007_3_1.py`](code/stock2007_3_1.py), [`…_modern.py`](code/stock2007_3_1_modern.py) | §3 UC-SV (stochastic volatility) model via Stan; smoothed σ_η, σ_ε and implied θ_t (Figure 2). |
| [`stock2007_4_1.py`](code/stock2007_4_1.py), [`…_modern.py`](code/stock2007_4_1_modern.py) | §4 / Table 4: relative MSFEs of AR(AIC), UC-SV and fixed-θ IMA(1,1). |

---

## Repository structure

```
.
├── README.md
├── code/                       # Python replication scripts
│   ├── sims1980_var.py         #   flagship CLI VAR/SVAR replication
│   ├── stock2007_*.py          #   Stock & Watson (2007) suite
│   ├── *_replication.py        #   per-paper scripts
│   └── data/                   #   local datasets used by scripts
├── pdf/
│   ├── vansteenberghe_MP.tex   # full LaTeX write-up (derivations + notes)
│   └── vansteenberghe_MP.pdf   # compiled paper
├── stan/                       # Stan models for the UC-SV estimation
│   ├── sw_ucsv.stan
│   └── sw_ucsv_gamma02.stan
└── data/                       # cached / downloaded datasets
```

---

## Data sources

Scripts fetch data directly from public sources where possible, and otherwise
read local files under `data/`:

- **JST Macrohistory Database, Release 6** (`narrowm`, `money`, `cpi`, `rgdpbarro`,
  `stir`, `ltrate`, …) — auto-downloaded from [macrohistory.net](https://www.macrohistory.net/);
  see Jordà, Schularick & Taylor (2017).
- **FRED** (`GDPC1`, `GDPDEF`, `CPILFESL`, `BOGMBASEW`, `FEDFUNDS`, `DFEDTAR`,
  `PRS85006173` (labor share), `IPDNBS`, `GS10`, `TB3MS`, `PPIACO`, …) via
  `pandas-datareader` / direct CSV.
- **ECB Data Portal API** — EURIBOR, Deposit Facility Rate, HICP.
- **Federal Reserve FCI-G** index (financial conditions) — `fci_g_public_monthly_3yr.csv`.
- **BODACC** French legal-announcements open-data API.
- **Stock & Watson (2007) replication files** — archival `p_gdp.q47`, `urate.q47`.
- **Global Macro Database** (optional, for [`sims_1980_replication_GMD.py`](code/sims_1980_replication_GMD.py)).

---

## Setup

Python 3.10+ recommended.

```bash
pip install numpy pandas scipy statsmodels matplotlib \
            pandas-datareader requests stargazer
# For the Stock & Watson UC-SV models (stock2007_3_1*, stock2007_4_1*):
pip install cmdstanpy arviz
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
# Optional, only for the Global Macro Database variant:
pip install global-macro-data
```

> **Working directory caveat.** Several scripts call
> `os.chdir('/Users/skimeur/Mon Drive/MP/')` near the top. Edit that line to your
> local checkout (or run from the repo root and remove it) before executing.

---

## Running

Most scripts are run directly:

```bash
python code/taylor_1993_replication.py
python code/stock2007_4_1_modern.py
```

The flagship Sims (1980) replication is a small CLI:

```bash
python code/sims1980_var.py --country USA --maxlags 8 --ic aic --prefer dta --horizon 12
```

Use `--no-cache` to force a fresh JST data download.

---

## Highlight — Sims (1980) VAR / SVAR

[`code/sims1980_var.py`](code/sims1980_var.py) replicates the empirical spirit of
**Sims (1980), *Macroeconomics and Reality*** using annual JST Macrohistory data for
`narrowm` (M1 proxy), `rgdpbarro` (output), `unemp`, `wage`, `cpi`, and `imports`
(import-price proxy). The baseline recursive ordering is
`narrowm, rgdpbarro, unemp, wage, cpi, imports`.

It implements:

- direct JST Release 6 download into a local `data/` cache,
- a reduced-form VAR with information-criterion lag selection,
- recursive (Cholesky) identification for orthogonalized impulse responses,
- forecast-error variance decomposition (FEVD),
- Granger / block-exogeneity tests (incl. money → unemployment, money → inflation),
- residual-whiteness and stability diagnostics,
- a robustness check across alternative recursive orderings.

**Methodological caveats.** A reduced-form VAR is a statistical forecasting system,
not a structural model on its own; structural interpretation requires the maintained
identification assumption, and recursive identification makes the variable ordering
part of that hypothesis — hence the ordering-robustness check. This is an
approximation with annual data, not the exact frequency/sample of Sims (1980).

---

## Author & citation

**Eric Vansteenberghe.** If you use this material, please cite the accompanying
write-up ([`pdf/vansteenberghe_MP.pdf`](pdf/vansteenberghe_MP.pdf)) and the original
papers it replicates.
