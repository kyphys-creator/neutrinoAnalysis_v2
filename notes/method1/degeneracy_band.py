"""Method 1: best fit + 縮退バンド (chi2 最小を保つ x_j の min/max) の図.
論文 Fig 11 の 'Degeneracy band' と同じ概念・様式.
usage: python degeneracy_band.py <1eV|5eV>
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import cvxpy as cp

REPO = '/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2'
THR  = sys.argv[1] if len(sys.argv) > 1 else '1eV'
EMIN = {'1eV': 0.18, '5eV': 0.41}[THR]
HERE = os.path.join(REPO, 'notes', 'method1')
RES  = os.path.join(HERE, 'results')

M1    = np.genfromtxt(os.path.join(HERE, 'data', f'CRmat_method1_{THR}_originalUnit.csv'), delimiter=',')
edges = np.genfromtxt(os.path.join(HERE, 'data', f'edges_method1_{THR}.csv'), delimiter=',')
n = len(edges) - 1

sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180',
                         GeV=0.32e16, solver='osqp', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
a.CRmat = M1*conv_mat; a.n = n; a.M_matrix = a.c*a.CRmat
a.data_vector = a.c*a.Ratebin7; a.Bkg_vector = np.zeros(a.m)
a._build_ordering_constraint()
a._dmb_default, a._inv_d_default = a._make_dmb_inv(a.data_vector)
a._hess_default = a._build_hessian_from(a._dmb_default, a._inv_d_default)
a._baseline_result = None; a.set_solver('osqp')
a._backend.vertex_select = True
res = a.optimize(a.data_vector)
conv = a.cm**2*a.sec
xbf = res.x*conv
print(f"[{THR}] best fit chi2/c = {res.fun/a.c:.2e}")

# ---- 縮退バンド: M_s x = mu_bf, 単調, 非負 のもとで各 x_j を min/max ----
Ms = a.M_matrix / a.c
mu = Ms @ res.x
x = cp.Variable(n, nonneg=True)
cons = [Ms @ x == mu, x[:-1] >= x[1:]]
lo, hi = np.array(xbf/conv), np.array(xbf/conv)
lo = lo.copy(); hi = hi.copy()
for j in range(n):
    for sense, arr in [(cp.Minimize, lo), (cp.Maximize, hi)]:
        prob = cp.Problem(sense(x[j]), cons)
        try:
            prob.solve(solver=cp.HIGHS, time_limit=15.0)
            if prob.status in ('optimal', 'optimal_inaccurate') and x.value is not None:
                arr[j] = x.value[j]
        except Exception:
            pass
lo, hi = lo*conv, hi*conv
width = (hi - lo)/xbf[0]
print(f"[band] max width / x(0) = {width.max():.3f};  wide (>1%) from "
      f"E = {edges[:-1][width > 0.01][0] if (width>0.01).any() else np.nan:.2f} MeV")

# ---- 図 (band_comparison 様式) ----
from scipy import integrate as _integrate
xg = np.logspace(-2, np.log10(7.0), 4000)
ys = np.interp(xg, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'])
yd = np.interp(xg, a.fig1dashed['MeV'], a.fig1dashed['cm**-2sec-1MeV-1'])
Phi_s = np.array([_integrate.trapezoid(ys[i:], xg[i:]) for i in range(len(xg))])
Phi_d = np.array([_integrate.trapezoid(yd[i:], xg[i:]) for i in range(len(xg))])

norm = 1e12
plt.style.use(os.path.join(REPO, '1eV', 'physrev.mplstyle'))
plt.figure(figsize=(8, 6))
plt.plot(xg, Phi_s/norm, color='black', lw=3, label='With NC')
plt.plot(xg, Phi_d/norm, color='black', lw=3, ls='dashed', label='Without NC')
Eplot = edges[:-1]
plt.fill_between(Eplot, lo/norm, hi/norm, step='post', color='#7030a0', alpha=0.25,
                 lw=0, label='Degeneracy band')
# 階段を一様 ~10 keV 間隔の点でなぞる (band 図と同じ見え方; 広い区間も点列で示す)
Es = np.arange(EMIN, 7.0, edges[1] - edges[0])
ystair = xbf[np.clip(np.searchsorted(edges, Es, side='right') - 1, 0, n - 1)]
plt.scatter(Es, ystair / norm, s=10, marker='o', color='#0072B2',
            zorder=5, label='No Bkg Best-fit')
plt.xscale('log'); plt.xlim(1e-1, 7.3)
plt.ylim(-0.06, 1.30*xbf[0]/norm)
plt.xlabel(r"$E_\nu$ [MeV]", fontsize=30)
plt.ylabel(nab._phi_ylabel(norm).replace('<~2~', '<~7~'), fontsize=30)
plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10)*0.1, numticks=20))
plt.tick_params(axis='both', which='major', labelsize=23)
plt.tick_params(axis='both', which='minor', labelsize=23)
plt.legend(loc='upper right', fontsize=15, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'method1_bestfit_degband_{THR}.pdf'))
plt.savefig(os.path.join(RES, f'method1_bestfit_degband_{THR}.png'), dpi=115)
json.dump(dict(threshold=THR, edges=edges.tolist(), best_fit=xbf.tolist(),
               deg_lo=lo.tolist(), deg_hi=hi.tolist()),
          open(os.path.join(RES, f'method1_degband_{THR}.json'), 'w'))
print("saved", f'method1_bestfit_degband_{THR}.pdf')
