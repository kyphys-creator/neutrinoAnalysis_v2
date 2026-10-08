# Bias / dispersion study (Monte-Carlo ensemble)

📖 [日本語](README.md)

Bias and variance of the point estimate requested by Bob Cousins, computed with the
procedure specified by Graciela.

**Procedure** — fluctuate the data $N_i$ predicted by the best fit with a Gaussian
$\mathcal{N}(N_i,\sqrt{N_i})$, refit each pseudo-dataset, and take the mean and
variance of $\delta\hat\Phi_j$. The truth is the best fit to the real data
($b_j=\langle\delta\hat\Phi_j\rangle-\delta\Phi_j^{\rm Best\,fit}$).

**Background treatment** — the figures in `results/` are the version with the
**background fluctuated as well** (the code's `bkg_penalty=True`):
$B_i=h_i+b_i$ is drawn from $\mathcal{N}(B_i,\sqrt{B_i})$ independently of the
pseudo-data and subtracted in the fit. The collaborators agreed the background
should be fluctuated. The initial version with the background fixed at its nominal
value is kept in `extra/bkg_fixed_figures/` for comparison.

> **For No Bkg the two versions are bit-for-bit identical.** With
> `background_scenario='none'` the `Bkg_vector` is exactly 0 in every bin, so there
> is nothing to fluctuate (not a bug). Only the two Exp Bkg configurations differ,
> with the dispersion increasing by about 12–13%.

Common settings: $\mathcal{E}=3$ kg·yr, $N_{\rm int}=80$, $M=500$, vertex
selection ON, seed 20260821.

## Folder layout

```
scripts/     rerun scripts (run_bias_variance.py, make_all_figures.py, make_histograms.py)
results/     figures for the four production configurations (what is sent to collaborators)
extra/       background-fixed figures, histograms, chi2
data/        raw ensemble output (JSON) and auxiliary data
archive/     old trial figures and logs (reference only, unused)
```

## results/ — the four production configurations

Each of `1eV_ExpBkg` / `1eV_NoBkg` / `5eV_ExpBkg` / `5eV_NoBkg` holds the same
six figure types (pdf+png):

| File | Content |
|---|---|
| `bias_{tag}_N80_vertexon` | bias $b_j$ (physical units) + $\pm\sigma_j$ band with I-type error bars |
| `relbias_reldisp_{tag}_N80` | relative bias $b_j/\delta\Phi_j^{\rm BF}$ and relative dispersion $\sigma_j/\delta\Phi_j^{\rm BF}$ in one figure |
| `relbias2_reldisp2_{tag}_N80` | squares of the above (log axis) |
| `bias2_over_var_{tag}_N80` | $b_j^2/\sigma_j^2$ (log axis, Tikhonov reference line = 1) |
| `hist_deltaPhi_{tag}_N80` | Bob's request: distribution of the point estimate $\delta\hat\Phi_j$ (5 energies) |
| `hist_residual_{tag}_N80` | Bob's request: distribution of $\delta\hat\Phi_j-\delta\Phi_j^{\rm true}$ (truth = input flux) |

The relative-quantity figures **exclude intervals with best fit = 0** (6 intervals
for 1 eV, 7 for 5 eV, all at the highest energies).

MC uncertainties: $\mathrm{Err}(b_j)=\sigma_j/\sqrt{M}$ (the error bars in the
figures), $\mathrm{Err}(\sigma_j^2)=\sigma_j^2\sqrt{2/(M-1)}=6.3\%$.
The variance uses $M-1$.

### Key numbers ($b_j^2/\sigma_j^2$)

Background-fluctuated version (`results/`; in parentheses the background-fixed
version `extra/bkg_fixed_figures/`):

| Configuration | $j$=0 | $j$=20 | $j$=45 | $j$=70 |
|---|---|---|---|---|
| 1eV ExpBkg | 0.455 (0.415) | 0.008 (0.008) | 0.005 (0.016) | 0.450 (0.582) |
| 1eV NoBkg  | 0.389 (same) | 0.015 (same) | 0.026 (same) | 0.689 (same) |
| 5eV ExpBkg | 0.380 (0.385) | 0.027 (0.053) | 0.043 (0.053) | 0.885 (1.098) |
| 5eV NoBkg  | 0.373 (same) | 0.096 (same) | 0.169 (same) | 1.007 (same) |

The bias is small at central energies and $\sim0.4$–$1.0$ (Tikhonov-like) at both
ends. Neither the background level (No Bkg / Exp Bkg) nor whether its uncertainty
is included makes much difference.

## extra/

- `histograms/` — initial version of the histograms (1eV ExpBkg, background fixed).
  The current version lives in `results/<tag>/`.
- `bkg_fixed_figures/` — figures with the background fixed at nominal
  (4 configurations × 4 figures). Paired with `results/` for comparison.
- `chi2_1eV_b_T3_N80_M500.csv` — $\chi^2$ of the 500 pseudo-experiments (both the
  background-fixed and fluctuated runs; same seed, so comparable replica by
  replica). **Values are divided by $10^4$.**

## data/

- `N80/` — JSONs of the production runs (the `_bkgvar` files are the
  background-fluctuated version = the source of `results/`). Contain `samples`
  (all 500×80 raw fit results), `mean`/`sd`/`bias`/`mse`/`corr`/`chi2`, etc.
- `N160/` — $N_{\rm int}=160$: M=500 (for the N80↔N160 comparison) and
  **M=10000** (the answer to Bob's "with $10^4$"; 4 configurations, same seed, so
  the first 500 toys coincide with the M=500 runs). Measured ~50–65 min per
  configuration (0.3–0.4 s/fit).
- `N180/` — $N_{\rm int}=180$: the early exploratory M=500 runs plus **M=10000**
  (4 configurations, seed 20260821, as a stability check against N160; medians
  agree with N160 within a few percent, only the extreme relative values grow).
- `true_deltaPhi_1eV_N80.npy` — true $\delta\Phi_j$ computed from the input flux
  (interval averages).
- `relative_1eV_N80.npz` — summary of the relative quantities for 1eV ExpBkg.

## Rerunning

```bash
cd notes/ensemble
PY=~/miniforge3/envs/python313/bin/python
DYLD_LIBRARY_PATH=~/miniforge3/envs/python313/lib $PY scripts/run_bias_variance.py <thr> <bkg> <T> <M> <seed> <modes> <N_int> [bkgvar]
```

- `<thr>`: `1eV` | `5eV`
- `<bkg>`: `b` (exponential) | `none` (no background) | `flat` | `a` | `c`
- `<modes>`: `vertex_on` | `vertex_off` | `vertex_on,vertex_off`
- passing `1` as the 8th argument also fluctuates the background

Example (the four production configurations):
```bash
$PY scripts/run_bias_variance.py 1eV b    3 500 20260821 vertex_on 80 1
$PY scripts/run_bias_variance.py 1eV none 3 500 20260821 vertex_on 80 1
$PY scripts/run_bias_variance.py 5eV b    3 500 20260821 vertex_on 80 1
$PY scripts/run_bias_variance.py 5eV none 3 500 20260821 vertex_on 80 1
$PY scripts/make_all_figures.py
$PY scripts/make_histograms.py
```

About 70 s per configuration. JSONs go to `data/N80/`, figures to
`results/<tag>/`. For the background-fixed version omit the 8th argument (the
input filenames in `make_all_figures.py` must be changed accordingly).

## Caveats

- **The vertex-selection LP fails for about 70% of the toys and falls back to the
  interior solution** (measured: 1eV ExpBkg, 100 toys: 76% at N80, 67% at N160).
  The cause: the QP interior solution violates monotonicity at the level of the
  solver tolerance, and is then passed to the LP as an exact equality constraint.
  `warnings.filterwarnings('ignore')` in `run_bias_variance.py` made this
  completely invisible before. The behavior has been common to all runs from the
  start, so the M=500 / M=10⁴ and N80 / N160 comparisons are consistent. In other
  words, the ensemble's "vertex ON" is in practice a mixture: a vertex for the
  ~30% of toys where the LP succeeds, the interior solution otherwise. For the
  best fit to the real data (the staircase in the paper) the LP succeeds.
- The vertex-selection LP occasionally enters a degenerate cycle inside the HiGHS
  simplex and **hangs** (first seen at 5eV N180; at N160 M=10⁴ it happened in 2 of
  4 runs). As a fix, every solver call in `_select_vertex` now has a **10-second
  time limit** (in both `1eV/` and `5eV/` copies of `neutrino_analysis_band.py`).
  A timed-out toy gets the same interior-solution fallback as above and leaves a
  `[vertex]` line in the log.
- Vertex selection ON/OFF affects the bias at the low-energy end (1eV ExpBkg,
  $j$=0: +104% vs +166%). For $j\gtrsim12$ they agree.
