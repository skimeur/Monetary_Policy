#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Euro-area New Keynesian Phillips curve: Galí-Gertler-López-Salido (GMM, lagged
instruments) versus Barnichon & Mesters (2020) shock-IV identification, with
weak-instrument-robust (subset Anderson-Rubin) inference.

The hybrid NKPC is estimated on the EURO-AREA LABOUR SHARE (real marginal cost),
    pi_t = gamma_b pi_{t-1} + gamma_f E_t pi_{t+1} + lambda s_t + e_t,
following the Barnichon-Mesters timing (E_t pi_{t+1} -> 4-quarter-ahead average
pi^F, pi_{t-1} -> 4-quarter-lagged average pi^L). On the common euro-area shock
window we estimate it two ways that differ ONLY in the instruments:

  (1) SHOCK-IV (Barnichon-Mesters): a sequence of euro-area PURE MONETARY shocks
      -- MP_median, Jarocinski & Karadi's median-rotation (structural) shock,
      taken unmodified from the authors' own repository
      github.com/marekjarocinski/jkshocks_update_ecb (monthly, 1999-2025) --
      and its H=20 lags, compressed to a quadratic (Almon) polynomial (three
      instruments).
  (2) LAGGED-IV control: four lags each of inflation and the labour share (the
      traditional Galí-Gertler / GGL instrument class), on the SAME sample.

Both report subset Anderson-Rubin confidence sets for lambda, gamma_f, gamma_b,
with the correct degrees of freedom (L - number of profiled nuisance regressors).

The question: do external euro-area monetary shocks deliver a BOUNDED confidence
set for the Phillips-curve slope where lagged instruments give an unbounded one?

@author: Eric Vansteenberghe
"""
import os
import sys
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.stats import chi2

os.chdir('/Users/skimeur/Mon Drive/MP/')
sys.path.insert(0, 'code')
import gali1999_euro_replication as ea     # euro-area data layer (pi, labour share)
import barnichon2020_nkpc as bm            # subset_AR, quad_spectral_kernel, GRID

OUT_DIR = 'output'
SHOCK_CSV = 'data/shocks_ecb_mpd_me_m.csv'  # JK authors' series: github.com/marekjarocinski/jkshocks_update_ecb
H = 20                                      # instrument lags (Barnichon-Mesters: 20)
GRID = bm.GRID

# context: published / our prior euro-area HYBRID estimates on the labour share
GGL_PUB = {'gamma_b': 0.025, 'gamma_f': 0.877, 'lambda': 0.018}   # GGL (2001) Table 2, xi=1, (1)
GGGMM_EA = {'gamma_b': 0.001, 'gamma_f': 0.989, 'lambda': 0.038}  # our gali1999_euro hybrid (1); AR unbounded
SHOCK_COL = 'MP_median'       # Jarocinski-Karadi (2020) pure-monetary shock (median rotation)


def load_mp_quarterly():
    sh = pd.read_csv(SHOCK_CSV)
    q = pd.PeriodIndex(pd.to_datetime(dict(year=sh.year, month=sh.month, day=1)), freq='Q')
    return sh.assign(q=q).groupby('q')[SHOCK_COL].sum()


def build_data():
    raw = ea.load_raw_euro()
    df = ea.make_variables(raw, 'GDPdef')                 # pi, s (=100*log labour share)
    mp = load_mp_quarterly()
    pi, s = df['pi'], df['s']
    piF = 0.25 * (pi.shift(-1) + pi.shift(-2) + pi.shift(-3) + pi.shift(-4))
    piL = 0.25 * (pi.shift(1) + pi.shift(2) + pi.shift(3) + pi.shift(4))
    cols = {'pi': pi, 'piL': piL, 'piF': piF, 's': s}
    for j in range(H + 1):                                # shock + 20 lags
        cols[f'z{j}'] = mp.shift(j)
    for j in range(1, 5):                                 # 4 lags of pi and s
        cols[f'piLag{j}'] = pi.shift(j)
        cols[f'sLag{j}'] = s.shift(j)
    return pd.DataFrame(cols).dropna()


def estimate():
    obs = build_data()
    T = len(obs)
    y = (obs['pi'] - obs['pi'].mean()).to_numpy()
    W = (obs[['piL', 'piF', 's']] - obs[['piL', 'piF', 's']].mean()).to_numpy()
    yT = y - W[:, 0]                                       # restricted: pi - piL
    WT = np.column_stack([W[:, 1] - W[:, 0], W[:, 2]])     # [piF - piL, s]

    max_lag = int(np.floor(4 * (T / 100) ** (2 / 9))) + 1
    K = bm.quad_spectral_kernel(T, max_lag)

    # instrument sets
    Zsh_raw = obs[[f'z{j}' for j in range(H + 1)]].to_numpy()
    r = np.arange(H + 1, dtype=float)
    Zsh = np.column_stack([Zsh_raw.sum(1), (Zsh_raw * r).sum(1), (Zsh_raw * r**2).sum(1)])
    Zlag = obs[[f'piLag{j}' for j in range(1, 5)]
               + [f'sLag{j}' for j in range(1, 5)]].to_numpy()
    Zlag = Zlag - Zlag.mean(0)

    out = {'T': T, 'span': f"{obs.index[0]}-{obs.index[-1]}", 'max_lag': max_lag}
    for tag, Z in [('shock', Zsh), ('lagged', Zlag)]:
        L = Z.shape[1]
        Pz = Z @ inv(Z.T @ Z) @ Z.T
        Mz = np.eye(T) - Pz
        MzKMz = Mz @ K @ Mz

        # point estimate: just-identified IV (shock) or 2SLS (lagged)
        if L == W.shape[1]:
            delta = inv(Z.T @ W) @ (Z.T @ y)
        else:
            delta = inv(W.T @ Pz @ W) @ (W.T @ Pz @ y)

        def ar_obj(theta):
            u = yT - WT @ theta
            return (u @ Pz @ u) / ((u @ MzKMz @ u) / T)
        deltaR = minimize(ar_obj, delta[1:3], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-10,
                                   'maxiter': 5000}).x

        def ar_ci(yv, mV, mX, level=0.95):
            crit = chi2.ppf(level, L - mX.shape[1])       # subset-AR df = L - #nuisance
            vals = np.array([bm.subset_AR(p, yv, mV, mX, Pz, MzKMz) for p in GRID])
            keep = vals < crit
            if not keep.any():
                return (np.nan, np.nan)
            idx = np.where(keep)[0]
            return (GRID[idx[0]], GRID[idx[-1]])

        ci = {'gamma_b': ar_ci(y, W[:, 0:1], W[:, 1:3]),
              'gamma_f': ar_ci(y, W[:, 1:2], W[:, [0, 2]]),
              'lambda': ar_ci(y, W[:, 2:3], W[:, 0:2])}
        ciR = {'gamma_f': ar_ci(yT, WT[:, 0:1], WT[:, 1:2]),
               'lambda': ar_ci(yT, WT[:, 1:2], WT[:, 0:1])}
        out[tag] = {'L': L, 'delta': delta, 'deltaR': deltaR, 'ci': ci, 'ciR': ciR}
    return out


def bounded(ci):
    lo, hi = ci
    if np.isnan(lo):
        return "empty"
    open_lo, open_hi = lo <= GRID[0], hi >= GRID[-1]
    if open_lo and open_hi:
        return "UNBOUNDED"
    tag = "" if not (open_lo or open_hi) else " (open)"
    return f"[{lo:.3f}, {hi:.3f}]{tag}"


def report(res):
    print("\n" + "=" * 78)
    print(f"EURO-AREA NKPC on the labour share  |  sample {res['span']}  "
          f"(T={res['T']}, HAC lag={res['max_lag']})")
    print("=" * 78)
    print(f"\nContext (hybrid NKPC on the labour share):")
    print(f"  GGL (2001) published (xi=1)  gb={GGL_PUB['gamma_b']:+.3f} "
          f"gf={GGL_PUB['gamma_f']:+.3f} lam={GGL_PUB['lambda']:+.3f}")
    print(f"  our GG/GMM EA 1995-2025      gb={GGGMM_EA['gamma_b']:+.3f} "
          f"gf={GGGMM_EA['gamma_f']:+.3f} lam={GGGMM_EA['lambda']:+.3f}  [AR unbounded]")
    for tag, name in [('shock', f'SHOCK-IV  ({SHOCK_COL}, Jarocinski-Karadi shock)'),
                      ('lagged', 'LAGGED-IV (4 lags of pi, s) -- control')]:
        r = res[tag]
        gb, gf, lam = r['delta']
        print(f"\n{name}   [L={r['L']} instruments]")
        print(f"  gamma_b = {gb:+.3f}   AR95 {bounded(r['ci']['gamma_b'])}")
        print(f"  gamma_f = {gf:+.3f}   AR95 {bounded(r['ci']['gamma_f'])}")
        print(f"  lambda  = {lam:+.3f}   AR95 {bounded(r['ci']['lambda'])}")
        print(f"  restricted (gamma_b+gamma_f=1): gamma_f={r['deltaR'][0]:+.3f} "
              f"AR95 {bounded(r['ciR']['gamma_f'])} ; "
              f"lambda={r['deltaR'][1]:+.3f} AR95 {bounded(r['ciR']['lambda'])}")


def latex_table(res):
    rows = [
        ('GGL (2001) published', f"{GGL_PUB['gamma_b']:.3f}", f"{GGL_PUB['gamma_f']:.3f}",
         f"{GGL_PUB['lambda']:.3f}", 'lagged (GMM)', '--'),
        ('GG/GMM EA 1995--2025', f"{GGGMM_EA['gamma_b']:.3f}", f"{GGGMM_EA['gamma_f']:.3f}",
         f"{GGGMM_EA['lambda']:.3f}", 'lagged (GMM)', 'unbounded'),
    ]
    lines = [r"\begin{tabular}{lcccll}", r"\toprule",
             r"method & $\gamma_b$ & $\gamma_f$ & $\lambda$ & instruments & "
             r"$95\%$ AR set for $\lambda$ \\", r"\midrule"]
    for m, gb, gf, lam, instr, arset in rows:
        lines.append(f"{m} & {gb} & {gf} & {lam} & {instr} & {arset} \\\\")
    for tag, name in [('shock', 'Barnichon shock-IV'),
                      ('lagged', 'lagged-IV (same sample)')]:
        r = res[tag]
        gb, gf, lam = r['delta']
        instr = 'JK monetary shock' if tag == 'shock' else 'lags of $\\pi,s$'
        lines.append(f"{name} & {gb:.3f} & {gf:.3f} & {lam:.3f} & {instr} & "
                     f"{bounded(r['ci']['lambda']).replace('UNBOUNDED','unbounded')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    res = estimate()
    report(res)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'barnichon2020_euro_compare.tex'), 'w') as f:
        f.write(latex_table(res))
    print(f"\nLaTeX written to {OUT_DIR}/barnichon2020_euro_compare.tex")


if __name__ == '__main__':
    main()
