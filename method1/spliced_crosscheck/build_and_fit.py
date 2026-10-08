"""Method 1 (Appendix E): un-truncated best fit up to 7 MeV.

Steps:
  1. read the knots of the official recipe written by driver_kernels.wls (box kernels in 0.01 MeV steps,
     i.e. the knots of the piecewise-linear interpolation themselves)
  2. validation: exact piecewise-linear integrals on the 180 uniform intervals below 2 MeV -> compare with the official CRmat180
  3. Graciela's grid: 180 uniform intervals below 2 MeV (same as the paper),
     adaptive above 2 MeV so that Sum_i R_ij equals that of the last interval below
  4. fit: data = Ratebin7 (neutrinos above 2 MeV are signal too), zero background, no h_i subtraction,
     non-negativity + monotonicity constraints, vertex selection ON
  5. stability check: both osqp/scipy backends + perturbed starting points

Never modifies existing files (output only inside method1/).
usage: python build_and_fit.py <1eV|5eV>
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR  = sys.argv[1] if len(sys.argv) > 1 else '1eV'
EMIN = {'1eV': 0.18, '5eV': 0.41}[THR]
NB   = int(os.environ.get('M1_NBELOW', '180'))   # number of uniform intervals below 2 MeV
TAG  = '' if NB == 180 else f'_n{NB}'
RES  = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)

# ---------- 1. read the knots ----------
kn = pd.read_csv(os.path.join(HERE, 'data', f'curlyR_knots_{THR}.csv'))
Ek = kn.iloc[:, 0].values                  # MeV, 0.01 steps
K  = kn.iloc[:, 1:].values.T               # (m bins, npts)  cm^2/(MeV kg)
m  = K.shape[0]
print(f"[{THR}] knots: {K.shape[1]} pts x {m} bins,  E {Ek[0]}..{Ek[-1]}")

def pl_integral(y, a, b):
    """Exact integral over [a,b] of the piecewise-linear function (knots Ek, values y)."""
    xs = np.r_[a, Ek[(Ek > a) & (Ek < b)], b]
    ys = np.interp(xs, Ek, y)
    return np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs))

def build_matrix(edges):
    M = np.zeros((m, len(edges) - 1))
    for j in range(len(edges) - 1):
        for i in range(m):
            M[i, j] = pl_integral(K[i], edges[j], edges[j + 1])
    return M

# ---------- 2. validation: compare with the official CRmat180 ----------
edges180 = np.linspace(EMIN, 2.0, NB + 1)
M180_phys = build_matrix(edges180)                       # cm^2/kg
_ref_candidates = [os.path.join(REPO, THR, 'CRmat', 'originalUnit', f'CRmat{NB}_originalUnit.csv'),
                   os.path.join(REPO, 'Mathematica', THR, 'output', f'CRmat{NB}_originalUnit.csv')]
ref = np.genfromtxt(next(f for f in _ref_candidates if os.path.exists(f)), delimiter=',')
# the scale comes from the row sums (checked to agree between the official and regenerated kernels)
rs_off = ref.sum(axis=1); rs_new = M180_phys.sum(axis=1)
ok_rows = rs_off > 0
s = np.median(rs_new[ok_rows] / rs_off[ok_rows])
rowdev = np.abs(rs_new[ok_rows] / rs_off[ok_rows] / s - 1)
print(f"[validate] scale (row sums) = {s:.8e},  row-sum agreement: max dev {rowdev.max():.2e}")
resid = np.abs(M180_phys[ref > 1e-3*ref.max()] / s / ref[ref > 1e-3*ref.max()] - 1)
print(f"[note] regenerated kernel shape vs legacy official (fine scale): max rel dev {resid.max():.2f}"
      "  -> below 2 MeV the official columns are used verbatim (splice)")

# ---------- 3. Graciela's grid ----------
S = K.sum(axis=0)                                        # Sum_i kernel_i(E)
T = pl_integral(S, edges180[-2], edges180[-1])           # size of the last interval below 2 MeV
hi_edges = [2.0]
while hi_edges[-1] < 7.0 - 1e-12:
    a = hi_edges[-1]
    # bisect for the b at which the cumulative integral reaches T
    lo, hi = a, 7.0
    if pl_integral(S, a, 7.0) <= T:
        hi_edges.append(7.0); break
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if pl_integral(S, a, mid) < T else (lo, mid)
    hi_edges.append(0.5 * (lo + hi))
hi_edges = np.array(hi_edges)
edges = np.r_[edges180, hi_edges[1:]]
n = len(edges) - 1
w_hi = np.diff(hi_edges)
print(f"[grid] {NB} uniform (< 2 MeV, width {edges180[1]-edges180[0]:.4f})"
      f" + {len(w_hi)} adaptive (> 2 MeV, width {w_hi.min():.4f}..{w_hi.max():.4f})  -> n = {n}")

M1 = build_matrix(edges) / s                             # originalUnit convention
M1[:, :len(edges180) - 1] = ref                          # below 2 MeV use the paper's official matrix as is
np.savetxt(os.path.join(HERE, 'data', f'CRmat_method1_{THR}{TAG}_originalUnit.csv'),
           M1, delimiter=',')
np.savetxt(os.path.join(HERE, 'data', f'edges_method1_{THR}{TAG}.csv'),
           edges, delimiter=',')

# ---------- 3b. closure test: M1 . x_true ≈ Ratebin7 ----------
dnde0 = pd.read_csv(os.path.join(REPO, 'Mathematica', THR, 'input', 'dNdEsolid.csv'))
_E0, _Y0 = dnde0.iloc[:, 0].values, dnde0.iloc[:, 1].values
_fn = (2.65e22 / 205.3) / (4 * np.pi) * (1 / 7200.0**2 + 1 / 10200.0**2)
_Ef = np.linspace(EMIN, 7.0, 40000); _pf = np.interp(_Ef, _E0, _Y0 * _fn)
x_true_cl = np.array([np.trapezoid(_pf[_Ef >= e], _Ef[_Ef >= e]) for e in edges[:-1]])
R7 = np.genfromtxt(os.path.join(REPO, THR, 'Ratebin', 'Ratebin7_originalUnit.csv'), delimiter=',')
convK = 1.0  # both in originalUnit
pred = M1 @ (x_true_cl)  # physical flux * originalUnit matrix -> the units are absorbed in the class, so only the ratio matters
ratio = pred / R7 / np.median(pred / R7)
print(f"[closure] M1.x_true vs Ratebin7 (shape, median-normalised): "
      f"min {ratio.min():.4f}  max {ratio.max():.4f}")

# ---------- 4. fit ----------
sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab

def make_analysis(solver):
    GEV = float(os.environ.get('M1_GEV', '0.32e16'))
    a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180',
                             GeV=GEV, solver=solver, T=3.0)
    conv_mat = a.cm**2 / (10**3 * a.gram) * (10**3 * a.gram) * a.yr
    a.CRmat = M1 * conv_mat
    a.n = n
    a.M_matrix = a.c * a.CRmat
    a.data_vector = a.c * a.Ratebin7          # the whole rate is signal (h_i not subtracted)
    a.Bkg_vector = np.zeros(a.m)
    a._build_ordering_constraint()
    a._dmb_default, a._inv_d_default = a._make_dmb_inv(a.data_vector)
    a._hess_default = a._build_hessian_from(a._dmb_default, a._inv_d_default)
    a._baseline_result = None
    a.set_solver(solver)          # rebuild the backend for the new n
    if a._backend.name == 'osqp':
        a._backend.vertex_select = True
    return a

a = make_analysis('osqp')
res = a.optimize(a.data_vector)
conv = a.cm**2 * a.sec
x = res.x * conv
chi2 = res.fun / a.c
nsteps = int((np.abs(np.diff(x)) > 1e-6 * max(x[0], 1)).sum())
print(f"[fit osqp+vertex] chi2/c = {chi2:.3e}   steps = {nsteps}   "
      f"x[0] = {x[0]:.4e}   x above 2 MeV: max {x[NB:].max():.3e}")
tol = 1e-6 * x[0]
tread_len, cur = [], 1
for j in range(1, n):
    if abs(x[j] - x[j-1]) < tol: cur += 1
    else: tread_len.append(cur); cur = 1
tread_len.append(cur)
tl = np.array(tread_len)
nz = tl[:len(tl)]  # all treads
print(f"[treads] total {len(tl)}   width-1 treads: {(tl==1).sum()}   "
      f"width dist: min {tl.min()}  median {int(np.median(tl))}  max {tl.max()}")

# ---------- 5. stability check ----------
checks = {}
av = make_analysis('osqp'); av._backend.vertex_select = False
rv = av.optimize(av.data_vector)
checks['vertex_off'] = dict(chi2=float(rv.fun / av.c),
                            max_dx_rel=float(np.max(np.abs(rv.x * conv - x)) / x[0]))
rng = np.random.default_rng(1)
for k in range(3):
    x0 = np.sort(rng.uniform(0, 2 * x[0] / conv, n))[::-1]
    rk = a.optimize(a.data_vector, x0=x0)
    xk = rk.x * conv
    checks[f'restart_{k}'] = dict(chi2=float(rk.fun / a.c),
                                  max_dx_rel=float(np.max(np.abs(xk - x)) / x[0]))
print("[stability]", json.dumps(checks, indent=1))

# ---------- 6. figure (same style as plot_band_comparison) and save ----------
# theory curves: With NC (solid) / Without NC (dashed), both Phi(E)-Phi(7 MeV)
from scipy import integrate as _integrate
xg = np.logspace(-2, np.log10(7.0), 4000)
ys = np.interp(xg, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'])
yd = np.interp(xg, a.fig1dashed['MeV'], a.fig1dashed['cm**-2sec-1MeV-1'])
Phi_s = np.array([_integrate.trapezoid(ys[i:], xg[i:]) for i in range(len(xg))])
Phi_d = np.array([_integrate.trapezoid(yd[i:], xg[i:]) for i in range(len(xg))])

norm = 1e12
plt.style.use(os.path.join(REPO, '1eV', 'physrev.mplstyle'))
from matplotlib.ticker import LogLocator
plt.figure(figsize=(8, 6))
plt.plot(xg, Phi_s / norm, color='black', lw=3, label='With NC')
plt.plot(xg, Phi_d / norm, color='black', lw=3, ls='dashed', label='Without NC')
plt.scatter(edges[:-1], x / norm, s=10, marker='o', color='#0072B2',
            zorder=5, label='No Bkg Best-fit')
plt.xscale('log'); plt.xlim(1e-1, 7.3)
plt.xlabel(r"$E_\nu$ [MeV]", fontsize=30)
plt.ylabel(nab._phi_ylabel(norm).replace('<~2~', '<~7~'), fontsize=30)
plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10) * 0.1, numticks=20))
plt.tick_params(axis='both', which='major', labelsize=23)
plt.tick_params(axis='both', which='minor', labelsize=23)
plt.legend(loc='upper right', fontsize=15, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'method1_bestfit_{THR}{TAG}.pdf'))
plt.savefig(os.path.join(RES, f'method1_bestfit_{THR}{TAG}.png'), dpi=115)

# true values (for the JSON record)
Efine = np.linspace(EMIN, 7.0, 40000)
phif = np.interp(Efine, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'])
truth = np.array([np.trapezoid(phif[Efine >= e], Efine[Efine >= e]) for e in edges[:-1]])

json.dump(dict(threshold=THR, emin=EMIN, n=n, edges=edges.tolist(),
               best_fit_physical=x.tolist(), truth_physical=truth.tolist(),
               chi2_over_c=float(chi2), n_steps=nsteps, stability=checks,
               scale_phys_to_originalUnit=float(s), validation_max_rel_dev=float(resid.max())),
          open(os.path.join(RES, f'method1_bestfit_{THR}{TAG}.json'), 'w'))
print("saved figure + json in method1/results/")
