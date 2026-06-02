#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Galí & Gertler (1999) New Keynesian Phillips Curve — EURO AREA replication.

This applies the GMM estimation of Galí & Gertler (1999), "Inflation dynamics:
A structural econometric analysis" (JME 44, 195-222) to euro-area data, in the
spirit of:

  Galí, Gertler & López-Salido (2001), "European inflation dynamics",
  European Economic Review 45(7), 1237-1270.

Models (same as the US script):
  - Table 1: forward-looking NKPC   pi_t = lambda*s_t + beta*E_t{pi_{t+1}}
  - Table 2: hybrid NKPC            pi_t = lambda*s_t + gamma_f*E_t{pi_{t+1}}
                                            + gamma_b*pi_{t-1}
Real marginal cost s_t is the (log) labor income share.

Euro-area data (quarterly, ~1995Q1-latest; bounded by Eurostat EA accounts):
  - Inflation: GDP deflator = nominal/real GDP (FRED CPMNACSCAB1GQEA19 /
    CLVMNACSCAB1GQEA19); HICP (FRED CP0000EZ19M086NEST) as an alternative panel.
  - Labor share (marginal cost): compensation of employees D1 / GDP B1GQ
    (Eurostat namq_10_gdp, EA20), adjusted for the self-employed using total
    employment vs. employees (Eurostat namq_10_a10_e) when available.
  - Instruments z_t: const + 4 lags each of inflation, the labor share, a
    quadratically detrended output gap, the 10y-3m rate spread (FRED
    IRLTLT01EZM156N, IR3TIB01EZQ156N), wage inflation (compensation per
    employee), and commodity-price inflation (FRED PPIACO).

Estimation: two-step GMM with a 12-lag Newey-West HAC weighting matrix and the
same moment normalizations (18)/(19) and (27)/(28) as the US script.

@author: Eric Vansteenberghe
"""

import os
import argparse
from io import StringIO

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2

os.chdir('/Users/skimeur/Mon Drive/MP/')

CACHE_DIR = 'data/gali1999_euro_cache'
OUT_DIR = 'output'

INSTR_VARS = ['pi', 's', 'gap', 'spread', 'winfl', 'cinfl']
NLAGS = 4
HAC_LAGS = 12
GEO = 'EA20'

# Galí & Gertler (1999) published US estimates (GDP deflator), for comparison.
GG_US_T1 = {'m1': {'theta': 0.829, 'beta': 0.926, 'lambda': 0.047},
            'm2': {'theta': 0.884, 'beta': 0.941, 'lambda': 0.021}}
GG_US_T2 = {'m1': {'omega': 0.265, 'theta': 0.808, 'beta': 0.885,
                   'gamma_f': 0.682, 'gamma_b': 0.252, 'lambda': 0.037},
            'm2': {'omega': 0.486, 'theta': 0.834, 'beta': 0.909,
                   'gamma_f': 0.591, 'gamma_b': 0.378, 'lambda': 0.015}}


# ----------------------------------------------------------------------
# Data fetchers
# ----------------------------------------------------------------------
def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name + '.csv')


def fred_series(series_id, refresh=False):
    """FRED series via the fredgraph CSV endpoint -> quarterly-mean Series."""
    cache = _cache_path('fred_' + series_id)
    if (not refresh) and os.path.exists(cache):
        raw = pd.read_csv(cache)
    else:
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text))
        raw.to_csv(cache, index=False)
    dates = pd.to_datetime(raw[raw.columns[0]])
    val = pd.to_numeric(raw[series_id], errors='coerce')
    s = pd.Series(val.to_numpy(), index=pd.PeriodIndex(dates, freq='Q'),
                  name=series_id)
    return s.groupby(level=0).mean()


def eurostat_series(dataflow, key, refresh=False, start='1995-Q1'):
    """Eurostat SDMX-CSV -> quarterly Series (TIME_PERIOD like '1995-Q1')."""
    cache = _cache_path(f'estat_{dataflow}_{key.replace(".", "_")}')
    if (not refresh) and os.path.exists(cache):
        raw = pd.read_csv(cache)
    else:
        url = (f'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/'
               f'{dataflow}/{key}?format=SDMX-CSV&startPeriod={start}')
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text))
        raw.to_csv(cache, index=False)
    raw = raw[['TIME_PERIOD', 'OBS_VALUE']].dropna()
    per = pd.PeriodIndex(raw['TIME_PERIOD'].astype(str).str.replace('-Q', 'Q',
                         regex=False), freq='Q')
    s = pd.Series(pd.to_numeric(raw['OBS_VALUE'], errors='coerce').to_numpy(),
                  index=per).sort_index()
    return s.groupby(level=0).mean()


def load_raw_euro(refresh=False):
    """Fetch all euro-area inputs; compute the (adjusted) labor share."""
    raw = {}
    # FRED euro-area aggregates
    raw['rgdp'] = fred_series('CLVMNACSCAB1GQEA19', refresh)   # real GDP
    raw['ngdp'] = fred_series('CPMNACSCAB1GQEA19', refresh)    # nominal GDP
    raw['y10'] = fred_series('IRLTLT01EZM156N', refresh)       # 10y govt yield
    raw['m3'] = fred_series('IR3TIB01EZQ156N', refresh)        # 3m interbank
    raw['ppi'] = fred_series('PPIACO', refresh)                # commodities
    raw['hicp'] = fred_series('CP0000EZ19M086NEST', refresh)   # HICP index
    # Eurostat national accounts (current prices, SA): compensation & GDP
    d1 = eurostat_series('namq_10_gdp', f'Q.CP_MEUR.SCA.D1.{GEO}', refresh)
    b1 = eurostat_series('namq_10_gdp', f'Q.CP_MEUR.SCA.B1GQ.{GEO}', refresh)

    # Self-employment adjustment: scale by total employment / employees
    adjusted = False
    try:
        # Eurostat namq_10_a10_e dimension order: freq.unit.nace_r2.s_adj.na_item.geo
        emp = eurostat_series('namq_10_a10_e',
                              f'Q.THS_PER.TOTAL.SCA.EMP_DC.{GEO}', refresh)
        sal = eurostat_series('namq_10_a10_e',
                              f'Q.THS_PER.TOTAL.SCA.SAL_DC.{GEO}', refresh)
        share = (d1 / b1) * (emp / sal)
        raw['comp_per_head'] = d1 / sal
        adjusted = True
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] employment series unavailable ({exc}); "
              "using unadjusted labor share")
        share = d1 / b1
        raw['comp_per_head'] = d1
    raw['labor_share'] = share
    raw['_adjusted'] = adjusted
    print(f"  labor share: {'adjusted' if adjusted else 'unadjusted'} "
          f"(EA national accounts, {GEO})")
    return raw


def quad_detrend(x):
    y = x.dropna()
    t = np.arange(len(y), dtype=float)
    X = np.column_stack([np.ones_like(t), t, t * t])
    coef, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    return pd.Series(y.to_numpy() - X @ coef, index=y.index).reindex(x.index)


def make_variables(raw, inflation='GDPdef'):
    """Build the quarterly regression variables for one inflation measure."""
    if inflation == 'GDPdef':
        price = raw['ngdp'] / raw['rgdp']           # GDP deflator (ratio)
    else:                                            # 'HICP'
        price = raw['hicp']
    df = pd.DataFrame(index=raw['labor_share'].index)
    df['pi'] = 100.0 * np.log(price).diff()
    df['s'] = 100.0 * np.log(raw['labor_share'])
    df['gap'] = quad_detrend(100.0 * np.log(raw['rgdp']))
    df['spread'] = raw['y10'] - raw['m3']
    df['winfl'] = 100.0 * np.log(raw['comp_per_head']).diff()
    df['cinfl'] = 100.0 * np.log(raw['ppi']).diff()
    return df.dropna(how='all').sort_index()


def assemble(df, start, end, nlags=None):
    nlags = NLAGS if nlags is None else nlags
    d = df.copy()
    win = d.loc[start:end]
    d['pi'] = d['pi'] - win['pi'].mean()
    d['s'] = d['s'] - win['s'].mean()
    cols = {'pi': d['pi'], 's': d['s'],
            'piF': d['pi'].shift(-1), 'piL': d['pi'].shift(1)}
    zcols = []
    for v in INSTR_VARS:
        for L in range(1, nlags + 1):
            cols[f'{v}_l{L}'] = d[v].shift(L)
            zcols.append(f'{v}_l{L}')
    reg = pd.DataFrame(cols)
    reg['const'] = 1.0
    reg = reg.loc[start:end].dropna()
    return reg, ['const'] + zcols


# ----------------------------------------------------------------------
# GMM core (identical to the US script)
# ----------------------------------------------------------------------
def safe_inv(M):
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def newey_west(m, q=HAC_LAGS):
    T = m.shape[0]
    S = (m.T @ m) / T
    for j in range(1, q + 1):
        w = 1.0 - j / (q + 1.0)
        G = (m[j:].T @ m[:-j]) / T
        S += w * (G + G.T)
    return S


def num_jac(f, x, h=1e-5):
    x = np.asarray(x, float)
    f0 = np.atleast_1d(f(x))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        step = h * max(1.0, abs(x[i]))
        dx = np.zeros_like(x)
        dx[i] = step
        J[:, i] = (np.atleast_1d(f(x + dx)) - np.atleast_1d(f(x - dx))) / (2 * step)
    return J


def gmm_linear(y, X, Z, q=HAC_LAGS):
    T = len(y)
    A = Z.T @ X / T
    b = Z.T @ y / T
    W1 = safe_inv(Z.T @ Z / T)
    solve = lambda W: np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
    d = solve(W1)
    for _ in range(2):
        u = y - X @ d
        Si = safe_inv(newey_west(Z * u[:, None], q))
        d = solve(Si)
    V = safe_inv(A.T @ Si @ A) / T
    g = b - A @ d
    J = T * g @ Si @ g
    dof = Z.shape[1] - X.shape[1]
    return {'theta': d, 'se': np.sqrt(np.diag(V)), 'V': V, 'J': J, 'dof': dof,
            'pJ': 1 - chi2.cdf(J, dof)}


def gmm_nonlinear(resid_fn, Z, starts, bounds, q=HAC_LAGS):
    T, L = Z.shape
    gbar = lambda psi: Z.T @ resid_fn(psi) / T

    def best(W):
        obj = lambda p: (lambda g: g @ W @ g)(gbar(p))
        r = None
        for s0 in starts:
            cand = minimize(obj, s0, method='L-BFGS-B', bounds=bounds)
            if r is None or cand.fun < r.fun:
                r = cand
        return r.x

    psi = best(safe_inv(Z.T @ Z / T))
    for _ in range(2):
        Si = safe_inv(newey_west(Z * resid_fn(psi)[:, None], q))
        psi = best(Si)
    G = num_jac(gbar, psi)
    V = safe_inv(G.T @ Si @ G) / T
    g = gbar(psi)
    J = T * g @ Si @ g
    dof = L - len(psi)
    return {'theta': psi, 'se': np.sqrt(np.diag(V)), 'V': V, 'J': J, 'dof': dof,
            'pJ': 1 - chi2.cdf(J, dof)}


def delta(fn, psi, V):
    G = num_jac(fn, psi)
    return np.atleast_1d(fn(psi)), np.sqrt(np.diag(G @ V @ G.T))


def ar_confset_lambda(a, Z, grid=None, level=0.95, q=HAC_LAGS):
    """Weak-IV-robust (Anderson-Rubin / Stock-Wright S) confidence set for the
    marginal-cost slope lambda in pi = lambda*s + beta*E[pi_{t+1}]; the nuisance
    beta is profiled out by minimizing the continuously-updated S-statistic, and
    lambda0 is retained iff min_beta S <= chi2(L) (projection method)."""
    pi, s, piF = a['pi'], a['s'], a['piF']
    T, L = Z.shape
    crit = chi2.ppf(level, L)
    if grid is None:
        grid = np.round(np.linspace(-0.30, 0.40, 281), 4)

    def s_min(lam0):
        def s_of_beta(be):
            u = pi - lam0 * s - be * piF
            gbar = Z.T @ u / T
            Si = safe_inv(newey_west(Z * u[:, None], q))
            return T * gbar @ Si @ gbar
        return minimize_scalar(s_of_beta, bounds=(0.5, 1.05),
                               method='bounded').fun

    S = np.array([s_min(l) for l in grid])
    return {'grid': grid, 'S': S, 'in': S <= crit, 'crit': crit, 'L': L}


def ar_summary(ar):
    grid, inset = ar['grid'], ar['in']
    if not inset.any():
        return "empty (rejected at all lambda on the grid)"
    idx = np.where(inset)[0]
    txt = f"[{grid[idx.min()]:+.3f}, {grid[idx.max()]:+.3f}]"
    if inset[0]:
        txt += " open-below"
    if inset[-1]:
        txt += " open-above"
    if idx.max() - idx.min() + 1 != len(idx):
        txt += " [non-contiguous]"
    return txt


def make_arrays(reg, zcols):
    a = {k: reg[k].to_numpy() for k in ('pi', 's', 'piF', 'piL')}
    return a, reg[zcols].to_numpy()


def table1_residuals(a):
    pi, s, piF = a['pi'], a['s'], a['piF']
    return {
        'm1': lambda p: p[0] * pi - (1 - p[0]) * (1 - p[1] * p[0]) * s - p[0] * p[1] * piF,
        'm2': lambda p: pi - (1 - p[0]) * (1 - p[1] * p[0]) / p[0] * s - p[1] * piF,
        'm1_rb': lambda p: p[0] * pi - (1 - p[0]) * (1 - p[0]) * s - p[0] * piF,
        'm2_rb': lambda p: pi - (1 - p[0]) * (1 - p[0]) / p[0] * s - piF,
    }


def table2_residuals(a):
    pi, s, piF, piL = a['pi'], a['s'], a['piF'], a['piL']
    phi = lambda th, be, om: th + om * (1 - th * (1 - be))

    def m1(p):
        th, be, om = p
        return (phi(th, be, om) * pi - (1 - om) * (1 - th) * (1 - be * th) * s
                - be * th * piF - om * piL)

    def m2(p):
        th, be, om = p
        ph = phi(th, be, om)
        return (pi - (1 - om) * (1 - th) * (1 - be * th) / ph * s
                - be * th / ph * piF - om / ph * piL)

    return {'m1': m1, 'm2': m2,
            'm1_rb': lambda p: m1((p[0], 1.0, p[1])),
            'm2_rb': lambda p: m2((p[0], 1.0, p[1])), 'phi': phi}


def run_reduced_form(a, Z):
    res = gmm_linear(a['pi'], np.column_stack([a['s'], a['piF']]), Z)
    (lam, be), (se_l, se_b) = res['theta'], res['se']
    return {'lambda': (lam, se_l), 'beta': (be, se_b),
            'J': res['J'], 'dof': res['dof'], 'pJ': res['pJ']}


def run_table1(a, Z):
    fns = table1_residuals(a)
    starts = [[t, b] for t in (0.6, 0.75, 0.85) for b in (0.90, 0.99)]
    starts_rb = [[t] for t in (0.6, 0.75, 0.85)]
    out = {}
    for key, restr in [('m1', False), ('m2', False), ('m1_rb', True), ('m2_rb', True)]:
        r = gmm_nonlinear(fns[key], Z, starts_rb if restr else starts,
                          [(0.05, 0.98)] if restr else [(0.05, 0.98), (0.80, 1.00)])
        psi, V = r['theta'], r['V']
        if restr:
            theta, beta = (psi[0], r['se'][0]), (1.0, 0.0)
            lam_fn = lambda p: (1 - p[0]) * (1 - p[0]) / p[0]
        else:
            theta, beta = (psi[0], r['se'][0]), (psi[1], r['se'][1])
            lam_fn = lambda p: (1 - p[0]) * (1 - p[1] * p[0]) / p[0]
        lam_v, lam_se = delta(lam_fn, psi, V)
        out[key] = {'theta': theta, 'beta': beta, 'lambda': (lam_v[0], lam_se[0]),
                    'J': r['J'], 'dof': r['dof'], 'pJ': r['pJ']}
    return out


def run_table2(a, Z):
    fns = table2_residuals(a)
    phi = fns['phi']
    starts = [[t, b, o] for t in (0.70, 0.83) for b in (0.90, 0.99) for o in (0.1, 0.3, 0.5)]
    starts_rb = [[t, o] for t in (0.70, 0.83) for o in (0.1, 0.3, 0.5)]

    def implied(p):
        th, be, om = p
        ph = phi(th, be, om)
        return np.array([be * th / ph, om / ph, (1 - om) * (1 - th) * (1 - be * th) / ph])

    out = {}
    for key, restr in [('m1', False), ('m2', False), ('m1_rb', True), ('m2_rb', True)]:
        r = gmm_nonlinear(fns[key], Z, starts_rb if restr else starts,
                          [(0.05, 0.98), (1e-3, 0.95)] if restr
                          else [(0.05, 0.98), (0.80, 1.00), (1e-3, 0.95)])
        psi, V = r['theta'], r['V']
        if restr:
            theta, omega, beta = (psi[0], r['se'][0]), (psi[1], r['se'][1]), (1.0, 0.0)
            imp_fn = lambda p: implied((p[0], 1.0, p[1]))
        else:
            theta, beta, omega = (psi[0], r['se'][0]), (psi[1], r['se'][1]), (psi[2], r['se'][2])
            imp_fn = implied
        iv, ise = delta(imp_fn, psi, V)
        out[key] = {'omega': omega, 'theta': theta, 'beta': beta,
                    'gamma_f': (iv[0], ise[0]), 'gamma_b': (iv[1], ise[1]),
                    'lambda': (iv[2], ise[2]),
                    'J': r['J'], 'dof': r['dof'], 'pJ': r['pJ']}
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def fmt(p, d=3):
    return f"{p[0]:.{d}f} ({p[1]:.{d}f})"


def estimate(raw, start, end, nlags=None):
    out = {'table1': {}, 'table2': {}}
    for infl in ('GDPdef', 'HICP'):
        df = make_variables(raw, infl)
        reg, zc = assemble(df, start, end, nlags)
        a, Z = make_arrays(reg, zc)
        if infl == 'GDPdef':
            out['reduced_form'] = run_reduced_form(a, Z)
            out['ar'] = ar_confset_lambda(a, Z)
            out['nobs'] = len(reg)
            out['span'] = f"{reg.index[0]}-{reg.index[-1]}"
        out['table1'][infl] = run_table1(a, Z)
        out['table2'][infl] = run_table2(a, Z)
    return out


def report(res):
    print("\n" + "=" * 80)
    print(f"GALI-GERTLER (1999) NKPC — EURO AREA ({GEO})  |  n={res['nobs']} "
          f"({res['span']})")
    print("=" * 80)
    rf = res['reduced_form']
    print(f"\nReduced form (GDP deflator):  lambda={fmt(rf['lambda'])}  "
          f"beta={fmt(rf['beta'])}  J p={rf['pJ']:.2f}")
    if 'ar' in res:
        print(f"  Anderson-Rubin 95% set for lambda (weak-IV robust, "
              f"L={res['ar']['L']} instr): {ar_summary(res['ar'])}")

    print("\nTable 1 — forward-looking NKPC        theta            beta          lambda     Jp")
    for infl in ('GDPdef', 'HICP'):
        for key, lab in [('m1', f'{infl} (1)'), ('m2', f'{infl} (2)')]:
            r = res['table1'][infl][key]
            print(f"  {lab:<12}{fmt(r['theta']):>16}{fmt(r['beta']):>16}"
                  f"{fmt(r['lambda']):>16}{r['pJ']:>6.2f}")

    print("\nTable 2 — hybrid NKPC      omega     theta      beta     gam_f     gam_b    lambda")
    for infl in ('GDPdef', 'HICP'):
        for key, lab in [('m1', f'{infl} (1)'), ('m2', f'{infl} (2)')]:
            r = res['table2'][infl][key]
            print(f"  {lab:<11}" + "".join(fmt(r[k]).rjust(10) for k in
                  ('omega', 'theta', 'beta', 'gamma_f', 'gamma_b', 'lambda')))


def report_vs_us(res):
    print("\n" + "=" * 80)
    print("COMPARISON — GDP deflator: Galí–Gertler US (1960–1997) vs Euro Area (this)")
    print("=" * 80)
    print("  Table 1                    theta     beta   lambda")
    for key in ('m1', 'm2'):
        g = GG_US_T1[key]
        e = res['table1']['GDPdef'][key]
        tag = '(1)' if key == 'm1' else '(2)'
        print(f"  GG US {tag}        {g['theta']:>9.3f}{g['beta']:>9.3f}{g['lambda']:>9.3f}")
        print(f"  Euro  {tag}        {e['theta'][0]:>9.3f}{e['beta'][0]:>9.3f}{e['lambda'][0]:>9.3f}")
    print("  Table 2          omega    theta     beta    gam_f    gam_b   lambda")
    for key in ('m1', 'm2'):
        g = GG_US_T2[key]
        e = res['table2']['GDPdef'][key]
        tag = '(1)' if key == 'm1' else '(2)'
        print(f"  GG US {tag}  " + "".join(f"{g[k]:>9.3f}" for k in
              ('omega', 'theta', 'beta', 'gamma_f', 'gamma_b', 'lambda')))
        print(f"  Euro  {tag}  " + "".join(f"{e[k][0]:>9.3f}" for k in
              ('omega', 'theta', 'beta', 'gamma_f', 'gamma_b', 'lambda')))


def latex_tables(res):
    t1 = [r"\begin{tabular}{lcccc}", r"\toprule",
          r" & $\theta$ & $\beta$ & $\lambda$ & $J$ $p$-val \\", r"\midrule"]
    for infl in ('GDPdef', 'HICP'):
        tag = 'GDP deflator' if infl == 'GDPdef' else 'HICP'
        t1.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{tag}}}}} \\")
        for key, lab in [('m1', '(1)'), ('m2', '(2)')]:
            r = res['table1'][infl][key]
            t1.append(f"{lab} & {fmt(r['theta'])} & {fmt(r['beta'])} & "
                      f"{fmt(r['lambda'])} & {r['pJ']:.2f} \\\\")
    t1 += [r"\bottomrule", r"\end{tabular}"]

    t2 = [r"\begin{tabular}{lcccccc}", r"\toprule",
          r" & $\omega$ & $\theta$ & $\beta$ & $\gamma_f$ & $\gamma_b$ & $\lambda$ \\",
          r"\midrule"]
    for infl in ('GDPdef', 'HICP'):
        tag = 'GDP deflator' if infl == 'GDPdef' else 'HICP'
        t2.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{tag}}}}} \\")
        for key, lab in [('m1', '(1)'), ('m2', '(2)')]:
            r = res['table2'][infl][key]
            t2.append(f"{lab} & {fmt(r['omega'])} & {fmt(r['theta'])} & "
                      f"{fmt(r['beta'])} & {fmt(r['gamma_f'])} & "
                      f"{fmt(r['gamma_b'])} & {fmt(r['lambda'])} \\\\")
    t2 += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(t1), "\n".join(t2)


def main():
    ap = argparse.ArgumentParser(description="Galí-Gertler (1999) NKPC — euro area")
    ap.add_argument('--refresh', action='store_true', help="re-download data")
    ap.add_argument('--nlags', type=int, default=NLAGS,
                    help="instrument lags per variable (default 4)")
    args = ap.parse_args()

    print("Loading euro-area data (FRED + Eurostat) ...")
    raw = load_raw_euro(refresh=args.refresh)
    start = str(raw['labor_share'].dropna().index[0])
    end = str(min(raw['rgdp'].dropna().index[-1], raw['labor_share'].dropna().index[-1]))

    res = estimate(raw, start, end, nlags=args.nlags)
    report(res)
    report_vs_us(res)

    os.makedirs(OUT_DIR, exist_ok=True)
    t1, t2 = latex_tables(res)
    with open(os.path.join(OUT_DIR, 'gali1999_euro_table1.tex'), 'w') as f:
        f.write(t1)
    with open(os.path.join(OUT_DIR, 'gali1999_euro_table2.tex'), 'w') as f:
        f.write(t2)
    print(f"\nLaTeX tables written to {OUT_DIR}/gali1999_euro_table*.tex")


if __name__ == '__main__':
    main()
