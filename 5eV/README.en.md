# neutrinoAnalysis

📖 [日本語](README.md)

Neutrino-flux optimization pipeline. The flux is reconstructed by χ² minimization
from the observed rates (`Ratebin`) and the response matrix (`CRmat`), and
Monte-Carlo (Neyman construction) confidence intervals are obtained for each flux
parameter.

Two interchangeable solver backends (`scipy` / `osqp`).

## Directory layout

```
neutrino_analysis_fast.py   # core; the scipy / osqp backends
neutrino_analysis_band.py   # fast + confidence-band root finding, saving, comparison plots
montecarlo.ipynb            # usage examples for optimization and scans
confidence_band.ipynb       # usage examples for band search, saving, overlays
confidence_band_flat/_Bkg/_noBkg.ipynb  # per-scenario band calculations
comparison.ipynb            # band comparison across scenarios
CRmat/originalUnit/         # response matrices CRmat<intervals>_originalUnit.csv
Ratebin/                    # Ratebin2 / Ratebin7 (observed rates)
Danny’s files/              # theory flux curves (fig1-solid / fig1-dashed etc.)
scenario_bkg_<x>/           # plot output (created at run time)
scenario_bkg_<x>/bands/     # confidence-band JSON output (created at run time)
```

These data folders must be present in the current working directory at run time.

## Dependencies

```bash
pip install numpy scipy matplotlib pandas numba
pip install cvxpy osqp        # required for solver='osqp'
pip install joblib            # optional: Monte-Carlo parallelism of the scipy backend
```

`cvxpy` also bundles CLARABEL / SCS / HiGHS (used for backend fallbacks and vertex
selection). Tested with Python 3.12.

## Basic usage

```python
from neutrino_analysis_fast import NeutrinoAnalysis

# scipy backend (stable, default)
a = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                     GeV=0.32e16, solver='scipy')

res = a.optimize(a.data_vector)      # χ² minimization
print(res.fun / a.c)                 # χ²/c
a.plot_flux_comparison(save=True)    # best-fit flux vs theory curves
```

### Switching solvers

```python
a.set_solver('osqp')   # fast backend (cvxpy + OSQP/CLARABEL/SCS)
```

- `scipy` — `trust-constr` with analytic Jacobian and constant Hessian. Stable.
- `osqp` — solves the χ² as a quadratic program. Fast for the free fit; the
  fixed-parameter fit switches internally to CLARABEL. In the underdetermined case
  (more parameters than bins) a HiGHS simplex picks a piecewise-constant
  (staircase) vertex solution (`_OSQPBackend.vertex_select`, default True).

Both backends return the same χ² definition. In the underdetermined case the flux
*shape* is not unique, so the backends may return different, equally optimal
solutions (see "Caveats" below).

## Δχ² scan (manual grid)

```python
a.optimize(a.data_vector)            # get the best fit first
scan = a.scan_fixed_parameter(
    fixed_index=39, scan_range=0.95, num_points=21,
    num_pseudo_data=100, seed=42, n_jobs=1,
)
import pandas as pd
pd.DataFrame(scan)                   # included / cutoff / Δχ² per fixed value
```

`scan` is a list of dicts; `pd.DataFrame(scan)` turns it directly into a table.

## Confidence-band root finding (`neutrino_analysis_band.py`)

Instead of tightening grid bounds by hand, bracket + geometric bisection locates
the band edges directly.

```python
from neutrino_analysis_band import NeutrinoAnalysis, load_band

a = NeutrinoAnalysis(background_scenario='flat', intervals='360',
                     GeV=0.32e16, solver='osqp')
a.optimize(a.data_vector)

# 1σ/90%/2σ simultaneously for one index
band = a.find_confidence_band(
    fixed_index=0,
    levels=(0.678, 0.90, 0.954),
    num_pseudo_data=20,   # bracketing stage (coarse)
    n_pseudo_edge=200,    # edge bisection (larger = more stable, slower)
    step=1.5, rel_tol=0.03, seed=42, verbose=True,
)
```

The widest level (2σ) brackets the outside; the edges of all levels nest inside
that interval. Only near the edges is `n_pseudo_edge` raised, to suppress the
Monte-Carlo noise of the cutoff. A fixed `seed` makes every point reproducible and
prevents bisection jitter.

### Saving several indices and overlaying them

```python
for idx in [0, 5, 10, 20, 40]:
    a.find_and_save_band(idx, outdir='scenario_bkg_flat/bands',
                         num_pseudo_data=50, n_pseudo_edge=500,
                         step=1.5, rel_tol=0.03, seed=42)

# overlay the saved bands on the optimize result
a.plot_flux_with_bands('scenario_bkg_flat/bands/band_*.json',
                       levels=(0.678, 0.90, 0.954), ylim=(0, 3e13))
```

- `save_band` / `load_band` … save/load one band per index as JSON
- `find_and_save_band` … search one index and save immediately
- `plot_flux_with_bands` … overlay saved bands as error bars per index on the
  scatter plot (the band center is `self.result.x[index]` at save time, so it
  coincides with the best-fit flux)

### Comparing bands across scenarios

```python
a_flat = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                          GeV=0.32e16, solver='osqp'); a_flat.optimize(a_flat.data_vector)
a_a    = NeutrinoAnalysis(background_scenario='a',    intervals='180',
                          GeV=0.32e16, solver='osqp'); a_a.optimize(a_a.data_vector)

a_flat.plot_band_comparison(
    {'flat': 'scenario_bkg_flat/bands/band_*.json',
     'a':    'scenario_bkg_a/bands/band_*.json'},
    level=0.954,
    optimized={'flat': a_flat, 'a': a_a},   # also overlay each scenario's optimize result
    ylim=(0, 3e13), save=True,
)
```

- Each scenario is overlaid with error bars in its own color. Bands are in
  physical units, so they can be compared regardless of the GeV unit choice.
- Passing `optimized={label: NeutrinoAnalysis or flux array}` overlays the
  best-fit flux scatter in the same color as the band (labels matching `groups`
  share the color).

## Guidance for the band-search parameters

| Argument | Role | Recommendation |
|---|---|---|
| `num_pseudo_data` | pseudo-datasets in the bracketing stage | enough to represent the requested percentile; **≥50** for 2σ |
| `n_pseudo_edge` | pseudo-datasets in the edge bisection (cutoff precision) | **≥500** (seed-dependent noise converges) |
| `step` | bracket growth factor (reach `v0·step^max_bracket`) | **1.5**. Too small (e.g. 1.05) with the default `max_bracket=25` cannot reach a distant upper edge and returns `inf` |
| `rel_tol` | relative edge tolerance (edge resolution) | 0.03. **Differences below this digit carry no meaning** |
| `seed` | fixed RNG (bisection reproducibility) | keep fixed |

Notes:
- The edge values carry Monte-Carlo fluctuations of the cutoff. With a small
  `n_pseudo_edge`, merely changing the seed moves the 2σ width by a few percent
  (90% is closer to the center and stabler; **2σ is an extreme quantile and
  especially jittery**).
- If a difference between scenarios is smaller than `rel_tol` (the edge
  resolution), its sign has no physical meaning. To claim a difference, **run a
  seed ensemble, quote edge ± error, and show the difference exceeds the error**.
- If the upper edge is only weakly constrained (a shallow profile), it is sounder
  to **report a one-sided limit** than to push the precision.

## Background scenarios

`background_scenario` is one of `'a' / 'b' / 'b2' / 'c' / 'flat' / 'none'`.
It can be switched at run time with `set_background()` (the OSQP cache is cleared
automatically).

## Caveats

- **Underdetermination**: e.g. `intervals='180'` has 180 flux parameters against
  29 observed bins (rank 29). The χ² minimum is a 151-dimensional face and the
  flux is not unique; scipy / osqp may return different equally optimal points.
  **The Δχ² profile (confidence intervals) depends only on χ² and is unaffected
  by this non-uniqueness**, so the physics conclusions are stable.
- **OSQP scaling**: the natural-unit constant `c` is huge (~1e70), so the problem
  is column-scaled internally before solving. Without this, OSQP/CLARABEL return a
  wrong point flagged `optimal`.
- **Band-search assumption**: the band is assumed connected (one crossing on each
  side).

## Backend differences (summary)

| | scipy | osqp |
|---|---|---|
| Solver | trust-constr | OSQP→CLARABEL→SCS (CLARABEL preferred for fixed-parameter fits) |
| Free fit | stable | fast |
| Fixed-parameter fit | stable | CLARABEL (OSQP fails to converge on large problems) |
| Monte-Carlo parallelism (`n_jobs`) | yes | no (sequential) |
| Flux shape | algorithm-dependent vertex | piecewise-constant vertex explicitly selected via HiGHS |
