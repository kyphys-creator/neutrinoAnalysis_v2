# Method 1 (Appendix E): un-truncated fit up to 7 MeV

📖 [日本語](README.md)

Subfolders organized by study. **Each subfolder is self-contained, using only its
own `data/` and `results/`** (shared inputs are duplicated per folder to keep them
independent). The only common references are to the repository itself (the analysis
classes in `1eV/` and `5eV/`, `Mathematica/*/output/curlyR_table.csv`, the official
CRmat files, and the physrev style).

> **Naming convention (2026-10-07)**: only `uniform/` (method1u) and `equalized/`
> (method1eq) may be called Method 1 — pure curlyR matrices with grid rules that
> never reference 2 MeV. `spliced_crosscheck/` (the former "method1" tag: a hybrid
> matrix with the official columns below 2 MeV) is a cross-check against the
> official matrices, not Method 1.

| Folder | Content |
|---|---|
| `mockdata/` | **Method 1 mock data (with resolution)**: `make_mockdata_res.py <thr>` applies curlyR_i (with the ±1 eV box resolution) to the theory flux (With NC) to build `data/Ratebin7res_<thr>_originalUnit.csv`, copied into `uniform/data` and `equalized/data`. The original Ratebin7 is resolution-free and inconsistent with curlyR_i (−3% in the 1–3 eV bin), so Method 1 uses this data instead (decided 2026-10-07). Contributions from E_ν < E_min are left out, following the paper's definition of E_ν^min (amount omitted: 5.0% of the 1–3 eV bin at 1 eV, 1.2% of the 5–7 eV bin at 5 eV) |
| `kernels/` | Box-kernel knots on the official 0.01 MeV grid, regenerated with Mathematica. `wolframscript -file driver_kernels.wls <1eV\|5eV>` → `data/curlyR_knots_*.csv`. The knots in other folders are copies of these |
| `uniform/` | **method1u**: pure curlyR matrix on a uniform grid (n=674/746). `build_uniform_Rij.py` → `fit_uniform_lp.py <thr>` (data defaults to `Ratebin7res`, `--data=ratebin7` for the original; LP fit + tail-weighted vertex + Δχ²=0 merge → d−1 steps) → `make_paper_figure.py <thr>` |
| `equalized/` | **method1eq**: adaptive grid refined so that no R_ij exceeds the largest element of the 2-eV-bin rows (n=920/1070). Same usage as uniform (tag method1eq); the result is the identical staircase to method1u. `compare_with_appendixC.py`: comparison with the best fits currently in Appendix C (spliced, N_int = 272/286, original data) — same number of steps d−1, mean difference below 2 MeV 2.8% / 2.6% (the largest, 16% / 10.5%, from where the first step is placed), both follow the truth to ~4% on average (`results/appC_vs_equalized.pdf`) |
| `soft_equalized/` | **method1soft** (trial, 2026-10-07): like equalized, refines where 𝓡_i is large, but keeps the peak shape instead of flattening it. With g(E) the column maximum at uniform width Δ, the width is w = Δ·min(1, (T/g)^α), so the refined maximum becomes T^α g^(1−α) (on a log axis the peak shape is simply compressed by 1−α). α = 1 reproduces equalized, α = 0 the uniform grid. Default α = 0.5: N_int = 764/862, largest element 2.20 T / 2.29 T (at 2.003 MeV, 81–120 eV bin). `build_soft_equalized_Rij.py <thr> [--alpha=…]`. Data: `Ratebin7res`. The best fit equals the equalized one (χ²=0, d−1 steps) |
| `soft_peakT/` | **method1softT** (trial, 2026-10-07): refinement of soft_equalized. For each row whose peak exceeds T (the wide bins), the width is chosen so that r_i = T (g_i/P_i)^(1−β), aligning the refined peaks with the 2-eV-bin peak T while keeping their shape (P_i = peak of row i at uniform width; β = 0 is similarity scaling, β = 1 flattens; default β = 0.5). The 2-eV bins are untouched. `build_soft_peakT_Rij.py <thr> [--beta=…]`. N_int = 1184/1423, refinement range 1.37–5.72 / 1.36–6.04 MeV, largest element 1.001 T / 1.002 T. Data: `Ratebin7res`. The best fit equals the equalized one (χ²=0, d−1 steps) |
| `fine_above_2MeV/` | **method1f** (trial, 2026-10-07): below 2 MeV = the same uniform 180 intervals as Method 2, above 2 MeV = a constant fine width (2.09 keV = Δ/4.85 at 1 eV, 1.68 keV = Δ/5.27 at 5 eV; N_int = 2577/3162). The width is the smallest number of intervals for which the above-2-MeV peak of the last bin's (81–120 eV) R_ij matches the first bin's peak (`build_fine_above2_Rij.py`). However, the last bin's curlyR_i peaks at 2.005 MeV and is nearly at its peak just below 2 MeV (width Δ), so the 1.99 MeV element remains 4.8× (1 eV) / 5.2× (5 eV) the first peak. Data: `Ratebin7res`. The best fit equals the equalized one (χ²=0, d−1 steps) |
| `resolution/` | Study of the inconsistency between the box (±1 eV resolution) kernels and the resolution-free data: `resolution_cut_check.py` (the −3% closure shift of the 1–3 eV bin = the cut at E_ν^min), `first_step_check.py` / `plot_first_step.py` (diagnostics of the first step), the `*_nores` matrices and their fits. **A study against the original data (Ratebin7, resolution-free); the box version's x(E_min) is ~10% above the nores version. Resolution: the data were rebuilt with resolution (`mockdata/`); with the box matrices + Ratebin7res the first step straddles the theory curve** |
| `spliced_crosscheck/` | The former "method1": official CRmat180 columns below 2 MeV + equal-integral adaptive grid above. cvxpy pipeline (`build_and_fit.py` etc.) and pure-scipy pipeline (`fit_and_merge_scipy.py`), d−1 merge (`exact_merge.py`), degeneracy band, tread-merge experiments, notebook. Its ~2% agreement with the nores version is the cross-check against the official matrices |
| `stability/` | Numerical stability and robustness experiments: `grid_dependence_test.py` (best fit unchanged for 46–184 intervals above 2 MeV), `stability_experiments.py` (the S_A adaptive / S_B uniform / S_C h_i-subtracted / S_D Method 2 dissection; results are printed only) |

Common physics conclusion: Method 1 is completely stable on a suitable grid
(χ²=0; the identical solution regardless of vertex selection or starting point;
d−1 = 30/28 steps, saturating the FDS bound). Above 2 MeV the fit is degenerate
(E_R^max(2 MeV) ≈ 120 eV = the upper edge of the recoil window), so only the total
is determined — the quantitative justification for cutting at 2 MeV in Method 2.
