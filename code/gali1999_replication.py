#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Galí and Gertler (1999), "Inflation dynamics: A structural econometric analysis",
Journal of Monetary Economics 44, 195-222.

Replication of the New Keynesian Phillips Curve (NKPC) estimates:
  - Table 1: pure forward-looking NKPC
        pi_t = lambda*s_t + beta*E_t{pi_{t+1}}                         (15)
  - Table 2: hybrid NKPC with rule-of-thumb price setters
        pi_t = lambda*s_t + gamma_f*E_t{pi_{t+1}} + gamma_b*pi_{t-1}   (26)

Marginal cost is measured by the (log) labor income share s_t (eq. 14).
Estimation is by GMM on the rational-expectations orthogonality conditions:
  Table 1: (18) "method 1" and (19) "method 2" normalisations of (17).
  Table 2: (27) "method 1" and (28) "method 2".
A 12-lag Newey-West HAC weighting matrix is used (paper's Tables 1-2 notes).

Instruments z_t: a constant plus four lags each of inflation, the labor share,
the output gap, the long-short interest-rate spread, wage inflation, and
commodity-price inflation.

Data: FRED, quarterly. The paper's sample is 1960Q1-1997Q4; we also run an
extended sample to the latest available quarter. Modern FRED vintages differ
from the paper's 1998-vintage data, so the estimates approximate (rather than
exactly reproduce) the published numbers.

@author: Eric Vansteenberghe
"""

import os
import argparse
from io import StringIO

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import chi2

# place yourself in the right directory (repo convention)
os.chdir('/Users/skimeur/Mon Drive/MP/')

CACHE_DIR = 'data/gali1999_cache'
OUT_DIR = 'output'

# FRED series and their native frequency ('Q' quarterly, 'M' monthly)
SERIES_FREQ = {
    'GDPDEF': 'Q',       # GDP implicit price deflator (overall)
    'IPDNBS': 'Q',       # Nonfarm business sector: implicit price deflator
    'PRS85006173': 'Q',  # Nonfarm business sector: labor share (index)
    'GDPC1': 'Q',        # Real GDP (for the output gap)
    'GS10': 'M',         # 10-year Treasury constant-maturity yield
    'TB3MS': 'M',        # 3-month Treasury bill, secondary market
    'PPIACO': 'M',       # PPI: all commodities
}
# Nonfarm business compensation per hour (wage inflation): try in order.
COMP_IDS = ['COMPNFB', 'PRS85006103']

INSTR_VARS = ['pi', 's', 'gap', 'spread', 'winfl', 'cinfl']
NLAGS = 4
HAC_LAGS = 12


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def fred_series(series_id, refresh=False):
    """Download one FRED series via the fredgraph CSV endpoint, with caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, series_id + '.csv')
    if (not refresh) and os.path.exists(cache):
        raw = pd.read_csv(cache)
    else:
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text))
        raw.to_csv(cache, index=False)
    date_col = raw.columns[0]
    dates = pd.to_datetime(raw[date_col])
    val = pd.to_numeric(raw[series_id], errors='coerce')  # '.' -> NaN
    s = pd.Series(val.to_numpy(), index=pd.PeriodIndex(dates, freq='Q'),
                  name=series_id)
    # collapse monthly observations to quarterly averages
    return s.groupby(level=0).mean()


def fred_first(ids, refresh=False):
    """Return the first FRED id in `ids` that downloads successfully."""
    last_err = None
    for sid in ids:
        try:
            return fred_series(sid, refresh=refresh), sid
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  [warn] could not fetch {sid}: {exc}")
    raise RuntimeError(f"none of {ids} could be fetched: {last_err}")


def load_raw(refresh=False):
    raw = {sid: fred_series(sid, refresh=refresh) for sid in SERIES_FREQ}
    comp, comp_id = fred_first(COMP_IDS, refresh=refresh)
    raw['COMP'] = comp
    print(f"  compensation series: {comp_id}")
    return raw


def quad_detrend(x):
    """Quadratic-detrend a series; return residual aligned to x's index."""
    y = x.dropna()
    t = np.arange(len(y), dtype=float)
    X = np.column_stack([np.ones_like(t), t, t * t])
    coef, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    resid = pd.Series(y.to_numpy() - X @ coef, index=y.index)
    return resid.reindex(x.index)


def make_variables(raw, deflator='GDP'):
    """Build the quarterly variables; inflation uses the chosen deflator."""
    price = raw['GDPDEF'] if deflator == 'GDP' else raw['IPDNBS']
    df = pd.DataFrame(index=price.index)
    df['pi'] = 100.0 * np.log(price).diff()                  # quarterly %, (15)
    df['s'] = 100.0 * np.log(raw['PRS85006173'])             # log labor share, (14)
    df['gap'] = quad_detrend(100.0 * np.log(raw['GDPC1']))   # detrended log GDP
    df['spread'] = raw['GS10'] - raw['TB3MS']                # long-short spread
    df['winfl'] = 100.0 * np.log(raw['COMP']).diff()         # wage inflation
    df['cinfl'] = 100.0 * np.log(raw['PPIACO']).diff()       # commodity inflation
    return df.sort_index()


def assemble(df, start, end):
    """
    Return (reg, zcols) restricted to [start, end] with complete cases.
    Inflation and the labor share are demeaned over the window (the model's
    zero-inflation steady state), so the intercept in z_t is innocuous and
    beta is not mechanically forced to one.
    """
    d = df.copy()
    win = d.loc[start:end]
    d['pi'] = d['pi'] - win['pi'].mean()
    d['s'] = d['s'] - win['s'].mean()

    cols = {'pi': d['pi'], 's': d['s'],
            'piF': d['pi'].shift(-1), 'piL': d['pi'].shift(1)}
    zcols = []
    for v in INSTR_VARS:
        for L in range(1, NLAGS + 1):
            name = f'{v}_l{L}'
            cols[name] = d[v].shift(L)
            zcols.append(name)
    reg = pd.DataFrame(cols)
    reg['const'] = 1.0
    zcols = ['const'] + zcols
    reg = reg.loc[start:end].dropna()
    return reg, zcols


# ----------------------------------------------------------------------
# GMM machinery
# ----------------------------------------------------------------------
def safe_inv(M):
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def newey_west(m, q=HAC_LAGS):
    """Bartlett (Newey-West) HAC covariance of the (T x L) moment series m."""
    T = m.shape[0]
    S = (m.T @ m) / T
    for j in range(1, q + 1):
        w = 1.0 - j / (q + 1.0)
        G = (m[j:].T @ m[:-j]) / T
        S += w * (G + G.T)
    return S


def num_jac(f, x, h=1e-5):
    """Central-difference Jacobian of vector function f at x (m x k)."""
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
    """Two-step efficient linear GMM of y = X delta, instruments Z."""
    T = len(y)
    A = Z.T @ X / T
    b = Z.T @ y / T
    W1 = safe_inv(Z.T @ Z / T)

    def solve(W):
        return np.linalg.solve(A.T @ W @ A, A.T @ W @ b)

    d = solve(W1)
    for _ in range(2):  # update HAC weight and re-estimate
        u = y - X @ d
        S = newey_west(Z * u[:, None], q)
        Si = safe_inv(S)
        d = solve(Si)
    V = safe_inv(A.T @ Si @ A) / T
    se = np.sqrt(np.diag(V))
    gbar = b - A @ d
    J = T * gbar @ Si @ gbar
    dof = Z.shape[1] - X.shape[1]
    return {'theta': d, 'se': se, 'V': V, 'J': J, 'dof': dof,
            'pJ': 1 - chi2.cdf(J, dof)}


def gmm_nonlinear(resid_fn, Z, starts, bounds, q=HAC_LAGS):
    """Two-step efficient nonlinear GMM with moment E[z_t * resid_fn(psi)] = 0."""
    T, L = Z.shape

    def gbar(psi):
        return Z.T @ resid_fn(psi) / T

    def best(W):
        objective = lambda p: (lambda g: g @ W @ g)(gbar(p))
        bestres = None
        for s0 in starts:
            r = minimize(objective, s0, method='L-BFGS-B', bounds=bounds)
            if bestres is None or r.fun < bestres.fun:
                bestres = r
        return bestres.x

    W1 = safe_inv(Z.T @ Z / T)
    psi = best(W1)
    for _ in range(2):
        u = resid_fn(psi)
        S = newey_west(Z * u[:, None], q)
        Si = safe_inv(S)
        psi = best(Si)
    G = num_jac(gbar, psi)
    V = safe_inv(G.T @ Si @ G) / T
    se = np.sqrt(np.diag(V))
    g = gbar(psi)
    J = T * g @ Si @ g
    dof = L - len(psi)
    return {'theta': psi, 'se': se, 'V': V, 'J': J, 'dof': dof,
            'pJ': 1 - chi2.cdf(J, dof)}


def delta(fn, psi, V):
    """Delta-method point estimate and SE for scalar/vector h = fn(psi)."""
    val = np.atleast_1d(fn(psi))
    G = num_jac(fn, psi)
    cov = G @ V @ G.T
    return val, np.sqrt(np.diag(cov))


# ----------------------------------------------------------------------
# Model residuals (demeaned arrays closed over)
# ----------------------------------------------------------------------
def make_arrays(reg, zcols):
    a = {k: reg[k].to_numpy() for k in ('pi', 's', 'piF', 'piL')}
    Z = reg[zcols].to_numpy()
    return a, Z


def table1_residuals(a):
    pi, s, piF = a['pi'], a['s'], a['piF']

    def m1(psi):  # normalisation (18): theta*pi - (1-theta)(1-beta*theta)s - theta*beta*piF
        th, be = psi
        return th * pi - (1 - th) * (1 - be * th) * s - th * be * piF

    def m2(psi):  # normalisation (19): pi - [(1-theta)(1-beta*theta)/theta] s - beta*piF
        th, be = psi
        lam = (1 - th) * (1 - be * th) / th
        return pi - lam * s - be * piF

    def m1_rb(psi):  # restricted beta = 1
        return m1((psi[0], 1.0))

    def m2_rb(psi):
        return m2((psi[0], 1.0))

    return {'m1': m1, 'm2': m2, 'm1_rb': m1_rb, 'm2_rb': m2_rb}


def table2_residuals(a):
    pi, s, piF, piL = a['pi'], a['s'], a['piF'], a['piL']

    def phi(th, be, om):
        return th + om * (1 - th * (1 - be))

    def m1(psi):  # normalisation (27)
        th, be, om = psi
        return (phi(th, be, om) * pi
                - (1 - om) * (1 - th) * (1 - be * th) * s
                - be * th * piF - om * piL)

    def m2(psi):  # normalisation (28)
        th, be, om = psi
        ph = phi(th, be, om)
        return (pi - (1 - om) * (1 - th) * (1 - be * th) / ph * s
                - be * th / ph * piF - om / ph * piL)

    def m1_rb(psi):  # restricted beta = 1, psi = (theta, omega)
        return m1((psi[0], 1.0, psi[1]))

    def m2_rb(psi):
        return m2((psi[0], 1.0, psi[1]))

    return {'m1': m1, 'm2': m2, 'm1_rb': m1_rb, 'm2_rb': m2_rb, 'phi': phi}


# ----------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------
def run_reduced_form(a, Z):
    """Linear GMM of (17): pi = lambda*s + beta*piF."""
    y = a['pi']
    X = np.column_stack([a['s'], a['piF']])
    res = gmm_linear(y, X, Z)
    (lam, be), (se_lam, se_be) = res['theta'], res['se']
    return {'lambda': (lam, se_lam), 'beta': (be, se_be),
            'J': res['J'], 'dof': res['dof'], 'pJ': res['pJ']}


def run_table1(a, Z):
    res_fns = table1_residuals(a)
    th_starts = [0.6, 0.75, 0.85]
    be_starts = [0.90, 0.99]
    starts = [[t, b] for t in th_starts for b in be_starts]
    starts_rb = [[t] for t in th_starts]
    bounds = [(0.05, 0.98), (0.80, 1.00)]  # beta <= 1 (admissible discount factor)
    bounds_rb = [(0.05, 0.98)]

    out = {}
    for key, restricted in [('m1', False), ('m2', False),
                            ('m1_rb', True), ('m2_rb', True)]:
        st = starts_rb if restricted else starts
        bd = bounds_rb if restricted else bounds
        r = gmm_nonlinear(res_fns[key], Z, st, bd)
        psi, V = r['theta'], r['V']
        if restricted:
            theta = (psi[0], r['se'][0]); beta = (1.0, 0.0)
            lam_fn = lambda p: (1 - p[0]) * (1 - p[0]) / p[0]
        else:
            theta = (psi[0], r['se'][0]); beta = (psi[1], r['se'][1])
            lam_fn = lambda p: (1 - p[0]) * (1 - p[1] * p[0]) / p[0]
        lam_v, lam_se = delta(lam_fn, psi, V)
        out[key] = {'theta': theta, 'beta': beta,
                    'lambda': (lam_v[0], lam_se[0]),
                    'J': r['J'], 'dof': r['dof'], 'pJ': r['pJ']}
    return out


def run_table2(a, Z):
    res_fns = table2_residuals(a)
    phi = res_fns['phi']
    starts = [[t, b, o] for t in (0.70, 0.83) for b in (0.90, 0.99)
              for o in (0.1, 0.3, 0.5)]
    starts_rb = [[t, o] for t in (0.70, 0.83) for o in (0.1, 0.3, 0.5)]
    bounds = [(0.05, 0.98), (0.80, 1.00), (1e-3, 0.95)]  # beta <= 1
    bounds_rb = [(0.05, 0.98), (1e-3, 0.95)]

    def implied(psi3):  # (gamma_f, gamma_b, lambda) from (theta, beta, omega)
        th, be, om = psi3
        ph = phi(th, be, om)
        gf = be * th / ph
        gb = om / ph
        lam = (1 - om) * (1 - th) * (1 - be * th) / ph
        return np.array([gf, gb, lam])

    out = {}
    for key, restricted in [('m1', False), ('m2', False),
                            ('m1_rb', True), ('m2_rb', True)]:
        st = starts_rb if restricted else starts
        bd = bounds_rb if restricted else bounds
        r = gmm_nonlinear(res_fns[key], Z, st, bd)
        psi, V = r['theta'], r['V']
        if restricted:
            theta = (psi[0], r['se'][0]); omega = (psi[1], r['se'][1])
            beta = (1.0, 0.0)
            imp_fn = lambda p: implied((p[0], 1.0, p[1]))
        else:
            theta = (psi[0], r['se'][0]); beta = (psi[1], r['se'][1])
            omega = (psi[2], r['se'][2])
            imp_fn = implied
        imp_v, imp_se = delta(imp_fn, psi, V)
        out[key] = {'omega': omega, 'theta': theta, 'beta': beta,
                    'gamma_f': (imp_v[0], imp_se[0]),
                    'gamma_b': (imp_v[1], imp_se[1]),
                    'lambda': (imp_v[2], imp_se[2]),
                    'J': r['J'], 'dof': r['dof'], 'pJ': r['pJ']}
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def fmt(pair, d=3):
    v, se = pair
    return f"{v:.{d}f} ({se:.{d}f})"


def print_block(title, rows, cols):
    print(f"\n{title}")
    head = "  {:<14}".format("") + "".join(f"{c:>16}" for c in cols)
    print(head)
    for label, vals in rows:
        line = "  {:<14}".format(label) + "".join(f"{v:>16}" for v in vals)
        print(line)


def report_sample(name, results):
    print("\n" + "=" * 78)
    print(f"GALI-GERTLER (1999) replication  --  sample: {name}")
    print(f"n = {results['nobs']} quarters ({results['span']})")
    print("=" * 78)

    rf = results['reduced_form']
    print("\nReduced-form NKPC (GDP deflator), eq. (17):  "
          f"pi = lambda*s + beta*E[pi(+1)]")
    print(f"  lambda = {fmt(rf['lambda'])}   beta = {fmt(rf['beta'])}   "
          f"J({rf['dof']}) = {rf['J']:.2f}, p = {rf['pJ']:.2f}")
    print("  (GG report lambda = 0.023 (0.012), beta = 0.942 (0.045))")

    # Table 1
    cols = ['theta', 'beta', 'lambda', 'J(p)']
    rows = []
    for defl in ('GDP', 'NFB'):
        t1 = results['table1'][defl]
        for key, lab in [('m1', f'{defl} (1)'), ('m2', f'{defl} (2)')]:
            r = t1[key]
            rows.append((lab, [fmt(r['theta']), fmt(r['beta']),
                               fmt(r['lambda']), f"{r['pJ']:.2f}"]))
        rb = t1['m1_rb']
        rows.append((f'{defl} restr.b1', [fmt(rb['theta']), '1.000',
                     fmt(rb['lambda']), f"{rb['pJ']:.2f}"]))
    print_block("Table 1 -- New Phillips curve (theta, beta, lambda):", rows, cols)
    print("  GG Table 1: GDP(1) theta=.829 beta=.926 lambda=.047 ; "
          "GDP(2) .884/.941/.021 ; NFB(1) .836/.957/.038")

    # Table 2
    cols = ['omega', 'theta', 'beta', 'gam_f', 'gam_b', 'lambda', 'p']
    rows = []
    for defl in ('GDP', 'NFB'):
        t2 = results['table2'][defl]
        for key, lab in [('m1', f'{defl} (1)'), ('m2', f'{defl} (2)')]:
            r = t2[key]
            rows.append((lab, [fmt(r['omega']), fmt(r['theta']), fmt(r['beta']),
                               fmt(r['gamma_f']), fmt(r['gamma_b']),
                               fmt(r['lambda']), f"{r['pJ']:.2f}"]))
        rb = t2['m1_rb']
        rows.append((f'{defl} restr.b1', [fmt(rb['omega']), fmt(rb['theta']),
                     '1.000', fmt(rb['gamma_f']), fmt(rb['gamma_b']),
                     fmt(rb['lambda']), f"{rb['pJ']:.2f}"]))
    print_block("Table 2 -- Hybrid Phillips curve (omega, theta, beta, ...):",
                rows, cols)
    print("  GG Table 2: GDP(1) om=.265 th=.808 be=.885 gf=.682 gb=.252 lam=.037")


def latex_table1(results, name):
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & $\theta$ & $\beta$ & $\lambda$ & $J$ $p$-val \\", r"\midrule"]
    for defl in ('GDP', 'NFB'):
        t1 = results['table1'][defl]
        tag = 'GDP deflator' if defl == 'GDP' else 'NFB deflator'
        lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{tag}}}}} \\")
        for key, lab in [('m1', '(1)'), ('m2', '(2)')]:
            r = t1[key]
            lines.append(f"{lab} & {fmt(r['theta'])} & {fmt(r['beta'])} & "
                         f"{fmt(r['lambda'])} & {r['pJ']:.2f} \\\\")
        rb = t1['m1_rb']
        lines.append(rf"restr. $\beta{{=}}1$ & {fmt(rb['theta'])} & 1.000 & "
                     f"{fmt(rb['lambda'])} & {rb['pJ']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_table2(results, name):
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & $\omega$ & $\theta$ & $\beta$ & $\gamma_f$ & $\gamma_b$ & "
             r"$\lambda$ \\", r"\midrule"]
    for defl in ('GDP', 'NFB'):
        t2 = results['table2'][defl]
        tag = 'GDP deflator' if defl == 'GDP' else 'NFB deflator'
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{tag}}}}} \\")
        for key, lab in [('m1', '(1)'), ('m2', '(2)')]:
            r = t2[key]
            lines.append(f"{lab} & {fmt(r['omega'])} & {fmt(r['theta'])} & "
                         f"{fmt(r['beta'])} & {fmt(r['gamma_f'])} & "
                         f"{fmt(r['gamma_b'])} & {fmt(r['lambda'])} \\\\")
        rb = t2['m1_rb']
        lines.append(rf"restr. $\beta{{=}}1$ & {fmt(rb['omega'])} & "
                     f"{fmt(rb['theta'])} & 1.000 & {fmt(rb['gamma_f'])} & "
                     f"{fmt(rb['gamma_b'])} & {fmt(rb['lambda'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Final comparison: GG (published) vs ours (same period) vs ours (longest)
# GDP deflator, methods (1) and (2) only -- the headline specifications.
# ----------------------------------------------------------------------
# Galí & Gertler (1999), Tables 1-2, GDP deflator columns (point estimates).
GG_T1 = {
    'm1': {'theta': 0.829, 'beta': 0.926, 'lambda': 0.047},
    'm2': {'theta': 0.884, 'beta': 0.941, 'lambda': 0.021},
}
GG_T2 = {
    'm1': {'omega': 0.265, 'theta': 0.808, 'beta': 0.885,
           'gamma_f': 0.682, 'gamma_b': 0.252, 'lambda': 0.037},
    'm2': {'omega': 0.486, 'theta': 0.834, 'beta': 0.909,
           'gamma_f': 0.591, 'gamma_b': 0.378, 'lambda': 0.015},
}
COMP_COLS = ['omega', 'theta', 'beta', 'gamma_f', 'gamma_b', 'lambda']


def _pt(x):
    """Point estimate from a (value, se) pair or a bare scalar."""
    return x[0] if isinstance(x, (tuple, list)) else x


def _comp_records(orig, ext):
    """Yield (spec, source_key, {col: value_or_None}) for the four key specs."""
    def t1(d):
        return {'omega': None, 'theta': _pt(d['theta']), 'beta': _pt(d['beta']),
                'gamma_f': None, 'gamma_b': None, 'lambda': _pt(d['lambda'])}
    def t2(d):
        return {k: _pt(d[k]) for k in COMP_COLS}
    for mk, spec in [('m1', 'Table 1, GDP deflator (1)'),
                     ('m2', 'Table 1, GDP deflator (2)')]:
        yield spec, 'gg', {**{c: None for c in COMP_COLS}, **GG_T1[mk]}
        yield spec, 'orig', t1(orig['table1']['GDP'][mk])
        yield spec, 'ext', t1(ext['table1']['GDP'][mk])
    for mk, spec in [('m1', 'Table 2, GDP deflator (1)'),
                     ('m2', 'Table 2, GDP deflator (2)')]:
        yield spec, 'gg', GG_T2[mk]
        yield spec, 'orig', t2(orig['table2']['GDP'][mk])
        yield spec, 'ext', t2(ext['table2']['GDP'][mk])


def comparison_latex(orig, ext, ext_lab):
    syms = {'omega': r'$\omega$', 'theta': r'$\theta$', 'beta': r'$\beta$',
            'gamma_f': r'$\gamma_f$', 'gamma_b': r'$\gamma_b$',
            'lambda': r'$\lambda$'}
    src = {'gg': r"Gal\'i--Gertler", 'orig': r"Ours 1960--1997",
           'ext': rf"Ours {ext_lab.replace('-', '--')}"}
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             " & & " + " & ".join(syms[c] for c in COMP_COLS) + r" \\",
             r"\midrule"]
    last = None
    for spec, s, vals in _comp_records(orig, ext):
        if spec != last:
            if last is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{spec}}}}} \\")
            last = spec
        cells = " & ".join("--" if vals[c] is None else f"{vals[c]:.3f}"
                           for c in COMP_COLS)
        lines.append(f" & {src[s]} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def report_comparison(orig, ext, ext_lab):
    src = {'gg': 'Gali-Gertler', 'orig': 'Ours 1960-1997',
           'ext': f'Ours {ext_lab}'}
    print("\n" + "=" * 78)
    print("FINAL COMPARISON  --  GDP deflator, methods (1) and (2)")
    print("=" * 78)
    print("  {:<28}{:<16}".format('spec', 'source')
          + "".join(f"{c:>9}" for c in ['omega', 'theta', 'beta',
                                        'gam_f', 'gam_b', 'lambda']))
    last = None
    for spec, s, vals in _comp_records(orig, ext):
        sp = spec if spec != last else ''
        last = spec
        cells = "".join(f"{vals[c]:>9.3f}" if vals[c] is not None
                        else f"{'--':>9}" for c in COMP_COLS)
        print("  {:<28}{:<16}".format(sp, src[s]) + cells)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def estimate_sample(raw, start, end):
    out = {'table1': {}, 'table2': {}}
    nobs = span = None
    for defl in ('GDP', 'NFB'):
        df = make_variables(raw, deflator=defl)
        reg, zcols = assemble(df, start, end)
        a, Z = make_arrays(reg, zcols)
        if defl == 'GDP':
            out['reduced_form'] = run_reduced_form(a, Z)
            nobs = len(reg)
            span = f"{reg.index[0]}-{reg.index[-1]}"
        out['table1'][defl] = run_table1(a, Z)
        out['table2'][defl] = run_table2(a, Z)
    out['nobs'] = nobs
    out['span'] = span
    return out


def main():
    ap = argparse.ArgumentParser(description="Galí-Gertler (1999) NKPC replication")
    ap.add_argument('--sample', choices=['original', 'extended', 'both'],
                    default='both')
    ap.add_argument('--refresh', action='store_true',
                    help="re-download FRED series (ignore cache)")
    args = ap.parse_args()

    print("Downloading / loading FRED data ...")
    raw = load_raw(refresh=args.refresh)
    last_q = min(s.dropna().index[-1] for s in raw.values())

    samples = []
    if args.sample in ('original', 'both'):
        samples.append(('original 1960Q1-1997Q4', '1960Q1', '1997Q4'))
    if args.sample in ('extended', 'both'):
        samples.append((f'extended 1960Q1-{last_q}', '1960Q1', str(last_q)))

    os.makedirs(OUT_DIR, exist_ok=True)
    collected = {}
    for name, start, end in samples:
        results = estimate_sample(raw, start, end)
        report_sample(name, results)
        tag = 'orig' if 'original' in name else 'ext'
        collected[tag] = results
        with open(os.path.join(OUT_DIR, f'gali1999_table1_{tag}.tex'), 'w') as f:
            f.write(latex_table1(results, name))
        with open(os.path.join(OUT_DIR, f'gali1999_table2_{tag}.tex'), 'w') as f:
            f.write(latex_table2(results, name))

    if 'orig' in collected and 'ext' in collected:
        ext_lab = '1960-' + collected['ext']['span'].split('-')[-1][:4]
        report_comparison(collected['orig'], collected['ext'], ext_lab)
        with open(os.path.join(OUT_DIR, 'gali1999_comparison.tex'), 'w') as f:
            f.write(comparison_latex(collected['orig'], collected['ext'], ext_lab))
    print(f"\nLaTeX tables written to {OUT_DIR}/gali1999_*.tex")


if __name__ == '__main__':
    main()
