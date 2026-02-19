# Monetary_Policy

Replication work for classic and modern monetary policy papers.

## Sims (1980) replication notes

This repository replicates the empirical spirit of **Sims (1980), _Macroeconomics and Reality_** with JST Macrohistory annual data for:

- `narrowm` (M1 proxy),
- `rgdpbarro` (real output proxy),
- `unemp`,
- `wage`,
- `cpi`,
- `imports` (import-price proxy).

Baseline ordering follows the paper's recursive identification spirit:
`narrowm, rgdpbarro, unemp, wage, cpi, imports`.

### What the script now implements

`./sims1980_var.py` now includes:

- direct JST Release 6 download from macrohistory.net into `./data` cache,
- reduced-form VAR with lag selection by IC,
- recursive (Cholesky) identification for orthogonalized IRFs,
- FEVD (forecast error variance decomposition),
- Granger/block-exogeneity-style tests (including money -> unemployment and money -> inflation),
- residual whiteness and stability diagnostics,
- simple robustness check to alternative recursive variable orderings.

### Run

```bash
python sims1980_var.py --country USA --maxlags 8 --ic aic --prefer dta --horizon 12
```

Use `--no-cache` to force a fresh data download.

### Methodological interpretation (short)

- The reduced-form VAR is a statistical forecast system, not by itself a deep structural model.
- Structural interpretation of shocks requires maintained identification assumptions.
- Recursive identification makes ordering part of the maintained hypothesis, so robustness checks are essential.

This implementation is an approximation with annual JST data, not the exact original frequency/sample used by Sims (1980).
