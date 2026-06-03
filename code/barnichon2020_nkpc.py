#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barnichon & Mesters (2020), "Identifying Modern Macro Equations with Old Shocks",
Quarterly Journal of Economics 135(4), 2255-2298.

Replication of TABLE I — the New Keynesian Phillips curve, Romer-Romer (2004)
monetary-shock identification, 1969-2007.

This is a faithful Python port of the authors' MATLAB replication code
(EmpiricalStudy/RomerRomer.m, fSubSet2.m, fCUErestricted.m). It estimates the
hybrid NKPC
    pi_t = gamma_b * pi_{t-1} + gamma_f * E_t{pi_{t+1}} + lambda * x_t + e_t,
where x_t is the unemployment gap or the output gap, by instrumental variables
in "impulse-response space": a sequence of Romer-Romer monetary policy shocks and
their H = 20 lags, compressed to a quadratic (Almon) polynomial in the lag
horizon (three instruments). Inference is weak-instrument robust, via the subset
Anderson-Rubin (min-eigenvalue) statistic; the restricted (gamma_b + gamma_f = 1)
point estimates are the continuously-updated estimator (CUE).

@author: Eric Vansteenberghe
"""

import os
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.linalg import toeplitz, cholesky
from scipy.optimize import minimize
from scipy.stats import chi2

os.chdir('/Users/skimeur/Mon Drive/MP/')

DATA = 'code/barnichon2020/ReplicationModernMacro/EmpiricalStudy/Data_QJE.xlsx'
OUT_DIR = 'output'
IL = 20                       # number of instrument lags H (RomerRomer.m: iL)
GRID = np.round(np.arange(-10.0, 10.0 + 1e-9, 0.01), 2)   # CI grid (-10:0.01:10)


# ----------------------------------------------------------------------
# Building blocks (ports of the MATLAB helpers)
# ----------------------------------------------------------------------
def quad_spectral_kernel(T, max_lag):
    """Quadratic-spectral HAC kernel as a Toeplitz weight matrix (RomerRomer.m
    lines 53-59). vQS[0]=1; vQS[l-1] for l=2..max_lag; zero beyond."""
    qs = np.zeros(T)
    qs[0] = 1.0
    for l in range(2, max_lag + 1):
        dX = (l - 1) / (1 + max_lag)
        z = 6 * np.pi * dX / 5
        qs[l - 1] = (25.0 / (12 * np.pi**2 * dX**2)) * (np.sin(z) / z - np.cos(z))
    return toeplitz(qs)


def subset_AR(p_fixed, y, mV, mX, Pz, MzKMz):
    """Subset Anderson-Rubin (min-eigenvalue) statistic — port of fSubSet2.m.

    Tests H0: the coefficients on mV equal p_fixed, profiling out the nuisance
    coefficients on mX. MATLAB:
        Ystar = [y - mV*p_fixed , mX]
        Rz    = Ystar' Mz K Mz Ystar / T
        AR    = min eig( inv(chol(Rz))' Ystar' Pz Ystar inv(chol(Rz)) ).
    numpy's cholesky is lower (L L' = Rz); MATLAB's is upper (R'R = Rz) with
    R = L', so inv(chol)' M inv(chol) = L^{-1} M L^{-T} (algebraically identical).
    Here MzKMz = Mz @ K @ Mz (the /T is applied below).
    """
    p_fixed = np.atleast_1d(np.asarray(p_fixed, float))
    Ystar = np.column_stack([y - mV @ p_fixed, mX])
    T = Ystar.shape[0]
    Rz = (Ystar.T @ MzKMz @ Ystar) / T
    M = Ystar.T @ Pz @ Ystar
    Linv = inv(cholesky(Rz, lower=True))
    return np.sort(np.linalg.eigvalsh(Linv @ M @ Linv.T))[0]


def ar_confidence_interval(y, mV, mX, Pz, MzKMz, T, level):
    """Invert the subset-AR test over GRID -> first/last grid value retained."""
    crit = chi2.ppf(level, 1)
    vals = np.array([subset_AR(p, y, mV, mX, Pz, MzKMz) for p in GRID])
    keep = vals < crit
    if not keep.any():
        return (np.nan, np.nan)
    idx = np.where(keep)[0]
    return (GRID[idx[0]], GRID[idx[-1]])


# ----------------------------------------------------------------------
# Estimation for one forcing variable
# ----------------------------------------------------------------------
def estimate(forcing):
    """forcing in {'ugap','ygap'}. Returns dict with point estimates + 95/90% CIs."""
    raw = pd.read_excel(DATA)
    pi = raw['PIX'].astype(float)                          # inflation (col 4)
    x = (raw['UR_hp'] if forcing == 'ugap' else raw['Ygap_hp']).astype(float)
    iv = raw['IV_rr'].astype(float)                        # Romer-Romer shock (col 7)

    # 4-quarter-ahead and 4-quarter-lagged average inflation (RomerRomer.m 28-29)
    piF = 0.25 * (pi.shift(-1) + pi.shift(-2) + pi.shift(-3) + pi.shift(-4))
    piL = 0.25 * (pi.shift(1) + pi.shift(2) + pi.shift(3) + pi.shift(4))

    # shock + 20 lags, then drop rows with any missing value (rmmissing)
    cols = {'pi': pi, 'piL': piL, 'piF': piF, 'x': x}
    for j in range(0, IL + 1):
        cols[f'z{j}'] = iv.shift(j)
    obs = pd.DataFrame(cols).dropna().reset_index(drop=True)
    T = len(obs)

    y = (obs['pi'] - obs['pi'].mean()).to_numpy()
    W = (obs[['piL', 'piF', 'x']] - obs[['piL', 'piF', 'x']].mean()).to_numpy()
    Z = obs[[f'z{j}' for j in range(IL + 1)]].to_numpy()    # T x 21 (raw shocks)

    # quadratic (Almon) polynomial instruments: [sum z, sum r*z, sum r^2 z]
    r = np.arange(IL + 1, dtype=float)
    Zp = np.column_stack([Z.sum(1), (Z * r).sum(1), (Z * r**2).sum(1)])

    # restricted (gamma_b + gamma_f = 1) variables
    yT = y - W[:, 0]                                        # pi - piL  (demeaned)
    WT = np.column_stack([W[:, 1] - W[:, 0], W[:, 2]])      # [piF - piL, x]

    # HAC kernel
    max_lag = int(np.floor(4 * (T / 100) ** (2 / 9))) + 1
    K = quad_spectral_kernel(T, max_lag)
    Pz = Zp @ inv(Zp.T @ Zp) @ Zp.T
    Mz = np.eye(T) - Pz
    MzKMz = Mz @ K @ Mz

    # unrestricted just-identified IV  ->  [gamma_b, gamma_f, lambda]
    delta = inv(Zp.T @ W) @ (Zp.T @ y)

    # restricted CUE: minimise the AR objective over [gamma_f, lambda]
    def ar_obj(theta):
        u = yT - WT @ theta
        num = u @ Pz @ u
        den = (u @ MzKMz @ u) / T
        return num / den
    res = minimize(ar_obj, delta[1:3], method='Nelder-Mead',
                   options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000})
    deltaR = res.x                                         # [gamma_f, lambda]

    out = {'forcing': forcing, 'T': T, 'max_lag': max_lag,
           'span': f"{raw['date'].iloc[0]:.2f}-{raw['date'].iloc[len(raw)-1]:.2f}",
           'delta': delta, 'deltaR': deltaR, 'ci': {}, 'ciR': {}}

    # ---- unrestricted confidence intervals (subset AR) ----
    specs = [('gamma_b', W[:, 0:1], W[:, 1:3]),            # fix gamma_b; nuisance piF,x
             ('gamma_f', W[:, 1:2], W[:, [0, 2]]),         # fix gamma_f; nuisance piL,x
             ('lambda',  W[:, 2:3], W[:, 0:2])]            # fix lambda;  nuisance piL,piF
    for name, mV, mX in specs:
        out['ci'][name] = {lvl: ar_confidence_interval(y, mV, mX, Pz, MzKMz, T, lvl)
                           for lvl in (0.95, 0.90)}

    # ---- restricted confidence intervals ----
    specsR = [('gamma_f', WT[:, 0:1], WT[:, 1:2]),         # fix gamma_f; nuisance x
              ('lambda',  WT[:, 1:2], WT[:, 0:1])]         # fix lambda;  nuisance piF-piL
    for name, mV, mX in specsR:
        out['ciR'][name] = {lvl: ar_confidence_interval(yT, mV, mX, Pz, MzKMz, T, lvl)
                            for lvl in (0.95, 0.90)}
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
PAPER = {  # published Table I, 95% CI, for side-by-side checking
    'ugap': {'gamma_b': (0.51, (0.11, 1.02)), 'gamma_f': (0.53, (0.07, 0.89)),
             'lambda': (-0.42, (-1.61, -0.05)),
             'gamma_f_R': (0.53, (0.11, 0.88)), 'lambda_R': (-0.45, (-1.57, -0.07))},
    'ygap': {'gamma_b': (0.62, (0.18, 3.31)), 'gamma_f': (0.42, (-2.05, 0.83)),
             'lambda': (0.28, (0.03, 2.95)),
             'gamma_f_R': (0.40, (-1.62, 0.82)), 'lambda_R': (0.31, (0.05, 2.53))},
}
LAB = {'ugap': 'U', 'ygap': 'Y'}


def report(r):
    f = r['forcing']
    p = PAPER[f]
    print(f"\n{'='*74}\nTABLE I — Phillips curve, RR id, forcing = {f}  "
          f"(T={r['T']}, HAC lag={r['max_lag']})\n{'='*74}")
    print(f"{'param':<9}{'ours est':>10}{'ours 95% CI':>22}"
          f"{'paper est':>11}{'paper 95% CI':>20}")
    gb, gf, lam = r['delta']
    rows = [('gamma_b', gb, r['ci']['gamma_b'][0.95], p['gamma_b']),
            ('gamma_f', gf, r['ci']['gamma_f'][0.95], p['gamma_f']),
            (f"lambda_{LAB[f]}", lam, r['ci']['lambda'][0.95], p['lambda'])]
    print("  -- unrestricted --")
    for name, est, ci, pap in rows:
        print(f"{name:<9}{est:>10.2f}{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>22}"
              f"{pap[0]:>11.2f}{f'[{pap[1][0]:.2f}, {pap[1][1]:.2f}]':>20}")
    print("  -- restricted (gamma_b+gamma_f=1) --")
    gfR, lamR = r['deltaR']
    rowsR = [('gamma_f', gfR, r['ciR']['gamma_f'][0.95], p['gamma_f_R']),
             (f"lambda_{LAB[f]}", lamR, r['ciR']['lambda'][0.95], p['lambda_R'])]
    for name, est, ci, pap in rowsR:
        print(f"{name:<9}{est:>10.2f}{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>22}"
              f"{pap[0]:>11.2f}{f'[{pap[1][0]:.2f}, {pap[1][1]:.2f}]':>20}")


def latex_table(results):
    def ci(t):
        return f"$[{t[0]:.2f},\\,{t[1]:.2f}]$"
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Unrestricted} & "
             r"\multicolumn{2}{c}{Restricted} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r" & & estimate & 95\% CI & estimate & 95\% CI \\", r"\midrule"]
    for r in results:
        f = r['forcing']
        tag = 'Unemployment gap' if f == 'ugap' else 'Output gap'
        gb, gf, lam = r['delta']
        gfR, lamR = r['deltaR']
        lab = LAB[f]
        lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{tag}}}}} \\")
        lines.append(rf"& $\gamma_b$ & {gb:.2f} & {ci(r['ci']['gamma_b'][0.95])} "
                     r"& -- & -- \\")
        lines.append(rf"& $\gamma_f$ & {gf:.2f} & {ci(r['ci']['gamma_f'][0.95])} "
                     rf"& {gfR:.2f} & {ci(r['ciR']['gamma_f'][0.95])} \\")
        lines.append(rf"& $\lambda_{{\mathrm{{{lab}}}}}$ & {lam:.2f} & "
                     rf"{ci(r['ci']['lambda'][0.95])} & {lamR:.2f} & "
                     rf"{ci(r['ciR']['lambda'][0.95])} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    results = [estimate('ugap'), estimate('ygap')]
    for r in results:
        report(r)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'barnichon2020_table1.tex'), 'w') as fh:
        fh.write(latex_table(results))
    print(f"\nLaTeX written to {OUT_DIR}/barnichon2020_table1.tex")


if __name__ == '__main__':
    main()
