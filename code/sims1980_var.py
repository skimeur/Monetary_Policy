#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sims (1980) inspired VAR/SVAR replication using JST Macrohistory data.

Paper:
Christopher A. Sims (1980), "Macroeconomics and Reality", Econometrica.

Replication notes also follow methodological guidance from:
- Helmut Lütkepohl (2005), New Introduction to Multiple Time Series Analysis.
- Kilian and Lütkepohl (2017), Structural Vector Autoregressive Analysis.
- Juselius (2006), The Cointegrated VAR Model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA_DIR = Path("data")
RAW_COLUMNS = ["year", "country", "narrowm", "rgdpbarro", "unemp", "wage", "cpi", "imports"]
PAPER_ORDER = ["narrowm", "rgdpbarro", "unemp", "wage", "cpi", "imports"]

# JST direct download links (Release 6)
JST_DTA_URL = "https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.dta?t=1720600177"
JST_XLSX_URL = "https://www.macrohistory.net/app/download/9834512569/JSTdatasetR6.xlsx?t=1720600177"


def _download(url: str, out_path: Path, overwrite: bool = False, timeout: int = 120) -> Path:
    """Download a file with streaming to avoid loading it all into RAM."""
    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
        tmp_path.replace(out_path)

    return out_path


def load_jst(cache: bool = True, prefer: str = "dta"):
    """Load JST data from the internet and optionally reuse local cache."""
    import pandas as pd

    prefer = prefer.lower().strip()
    if prefer not in {"dta", "xlsx"}:
        raise ValueError("prefer must be one of {'dta','xlsx'}")

    if prefer == "dta":
        path = DATA_DIR / "JSTdatasetR6.dta"
        _download(JST_DTA_URL, path, overwrite=not cache)
        return pd.read_stata(path)

    path = DATA_DIR / "JSTdatasetR6.xlsx"
    _download(JST_XLSX_URL, path, overwrite=not cache)
    return pd.read_excel(path)


def load_country_data(country: str = "USA", cache: bool = True, prefer: str = "dta"):
    """Load JST data, keep one country, and preserve Sims (1980) variable order."""
    data = load_jst(cache=cache, prefer=prefer)
    df = data.loc[data["country"] == country, ["year", "country", *PAPER_ORDER]].copy()
    return df.sort_values("year").set_index("year")


def build_var_dataset(df):
    """Create transformed annual series used in the baseline reduced-form VAR."""
    import numpy as np
    import pandas as pd

    out = pd.DataFrame(index=df.index)
    out["dlog_m"] = np.log(df["narrowm"] / df["cpi"]).diff()
    out["dlog_y"] = np.log(df["rgdpbarro"]).diff()
    out["unemp"] = df["unemp"]
    out["dlog_w"] = np.log(df["wage"]).diff()
    out["inflation"] = np.log(df["cpi"]).diff()
    out["dlog_imp"] = np.log(df["imports"]).diff()

    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    out.index = pd.PeriodIndex(out.index.astype(int), freq="Y")
    return out


def fit_var(df, maxlags: int = 8, ic: str = "aic"):
    """Select lag order by information criteria and fit reduced-form VAR."""
    from statsmodels.tsa.api import VAR

    model = VAR(df)
    selected = model.select_order(maxlags=maxlags, trend="c")
    lag = getattr(selected, ic)
    if lag is None or lag <= 0:
        lag = 1
    return selected, model.fit(lag, trend="c")


def report_diagnostics(results) -> None:
    """Print diagnostics recommended in standard VAR practice."""
    print("\n=== Stability check ===")
    print(results.is_stable(verbose=True))

    print("\n=== Residual whiteness test ===")
    try:
        print(results.test_whiteness(nlags=12))
    except Exception as exc:
        print(f"Whiteness test unavailable: {exc}")


def run_causality_tests(results) -> None:
    """Run Granger causality tests motivated by Sims (1980)."""
    print("\n=== Granger: narrow money growth -> unemployment ===")
    print(results.test_causality("unemp", ["dlog_m"], kind="f"))

    print("\n=== Granger: narrow money growth -> inflation ===")
    print(results.test_causality("inflation", ["dlog_m"], kind="f"))

    print("\n=== Granger: real activity (output, unemployment) -> money growth ===")
    print(results.test_causality("dlog_m", ["dlog_y", "unemp"], kind="f"))


def run_irf_fevd(results, horizon: int = 12) -> None:
    """Plot recursive-identified IRFs and FEVD."""
    import matplotlib.pyplot as plt

    irf = results.irf(horizon)
    irf.plot(orth=True)
    plt.suptitle("Orthogonalized IRFs (recursive/Cholesky identification)", y=1.02)
    plt.tight_layout()

    fevd = results.fevd(horizon)
    fevd.plot()
    plt.suptitle("Forecast Error Variance Decomposition", y=1.02)
    plt.tight_layout()
    plt.show()


def run_ordering_robustness(df_var, maxlags: int, ic: str) -> None:
    """Simple robustness check: compare baseline and alternative recursive orderings."""
    baseline = ["dlog_m", "dlog_y", "unemp", "dlog_w", "inflation", "dlog_imp"]
    alternative = ["dlog_y", "unemp", "inflation", "dlog_m", "dlog_w", "dlog_imp"]

    print("\n=== Ordering robustness check (recursive identification) ===")
    for order_name, ordering in [("baseline", baseline), ("alternative", alternative)]:
        _, res = fit_var(df_var[ordering], maxlags=maxlags, ic=ic)
        irf = res.irf(1)
        # contemporaneous response magnitude proxy: horizon 0 response of first equation to first shock
        impact = float(irf.orth_irfs[0, 0, 0])
        print(f"{order_name:>11} ordering | selected lag={res.k_ar} | own-impact(0)={impact:.6f}")


def run_pipeline(country: str, maxlags: int, ic: str, cache: bool, prefer: str, horizon: int) -> None:
    df_raw = load_country_data(country=country, cache=cache, prefer=prefer)
    df_var = build_var_dataset(df_raw)

    print("\n=== Replication variable mapping (Sims order in levels) ===")
    print("money->narrowm, real output->rgdpbarro, unemployment->unemp, wages->wage, prices->cpi, import prices proxy->imports")

    print("\n=== Preview transformed VAR dataset ===")
    print(df_var.head())

    selected, results = fit_var(df_var, maxlags=maxlags, ic=ic)
    print("\n=== Lag order selection ===")
    print(selected.summary())

    print("\n=== VAR estimation summary ===")
    print(results.summary())

    report_diagnostics(results)
    run_causality_tests(results)
    run_ordering_robustness(df_var, maxlags=maxlags, ic=ic)
    run_irf_fevd(results, horizon=horizon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sims (1980)-style VAR with JST data")
    parser.add_argument("--country", type=str, default="USA", help="Country code/name in JST data")
    parser.add_argument("--maxlags", type=int, default=8, help="Maximum lag order for selection")
    parser.add_argument("--horizon", type=int, default=12, help="IRF/FEVD horizon")
    parser.add_argument("--no-cache", action="store_true", help="Force redownload instead of using ./data cache")
    parser.add_argument(
        "--prefer",
        type=str,
        default="dta",
        choices=["dta", "xlsx"],
        help="Preferred JST source format",
    )
    parser.add_argument(
        "--ic",
        type=str,
        default="aic",
        choices=["aic", "bic", "hqic", "fpe"],
        help="Information criterion for lag selection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        country=args.country,
        maxlags=args.maxlags,
        ic=args.ic,
        cache=not args.no_cache,
        prefer=args.prefer,
        horizon=args.horizon,
    )
