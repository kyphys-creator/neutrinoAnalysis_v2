"""First step of the 1 eV fit: curlyR (box resolution) matrix vs no-resolution matrix.

For each matrix, the Delta chi2 = 0 band below 1 MeV (linprog) and the best fit
(from fit_uniform_lp.py) are overlaid on the true curve.

usage: python plot_first_step.py   (after fit_uniform_lp.py 1eV and 1eV nores)
output: results/first_step_compare_1eV.pdf/.png
"""
import os, sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, 'results')
ECUT = 1.0
sys.path.insert(0, os.path.join(REPO, '1eV')); os.chdir(os.path.join(REPO, '1eV'))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='scipy', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr; conv = a.cm**2*a.sec
data = np.asarray(a.Ratebin7, dtype=float)
edges = np.loadtxt(os.path.join(HERE, 'data', 'edges_method1u_1eV.csv'), delimiter=',')
n = len(edges)-1; XS = 1e12/conv
J = np.where(edges[:-1] < ECUT)[0]
Dm = (eye(n, n, k=1) - eye(n, n)).tocsr()[:-1]

def band(M):
    A = csr_matrix((M*conv_mat)*XS/data[:, None])
    out = np.zeros((len(J), 2))
    for k, j in enumerate(J):
        c = np.zeros(n); c[j] = 1.0
        f = lambda cc: linprog(cc, A_ub=Dm, b_ub=np.zeros(n-1), A_eq=A, b_eq=np.ones(len(data)),
                               bounds=(0, None), method='highs').fun
        out[k] = (f(c)*XS*conv, -f(-c)*XS*conv)
    return out

Ef = np.linspace(0.1, 7.2, 400001)
phif = np.interp(Ef, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'], right=0)
Phi = np.r_[np.cumsum((0.5*(phif[1:]+phif[:-1])*np.diff(Ef))[::-1])[::-1], 0.0]

CASES = [('method1u', r'$\mathcal{R}_i$ with $\pm1$ eV box resolution (current)'),
         ('method1u_nores', r'$\mathcal{R}_i$ without resolution (same as the mock data)')]
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(os.path.join(REPO, '1eV', 'physrev.mplstyle'))
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
N = 1e12; e_lo = edges[J[0]:J[-1]+2]
for ax, (tag, title), lab in zip(axes, CASES, ('(a)', '(b)')):
    M = np.loadtxt(os.path.join(HERE, 'data', f'CRmat_{tag}_1eV_originalUnit.csv'), delimiter=',')
    b = band(M)
    src = os.path.join(HERE, 'data') if tag == 'method1u' else RES
    x = np.load(os.path.join(src, f'{tag}_bestfit_1eV.npz'))['x']
    ax.stairs(b[:, 1]/N, e_lo, baseline=b[:, 0]/N, fill=True, color='#0072B2', alpha=0.18, lw=0,
              label=r'$\Delta\chi^2=0$ band')
    ax.plot(Ef, Phi/N, color='black', lw=2, label='true flux (with NC)')
    ax.stairs(x[J]/N, e_lo, baseline=None, color='#0072B2', lw=2.2, label='best fit')
    ax.set_xlim(0.15, ECUT); ax.set_ylim(0.9, 2.6)
    ax.set_xlabel(r'$E_\nu$ [MeV]', fontsize=22)
    ax.set_title(f'{lab} {title}', fontsize=14)
    ax.tick_params(labelsize=16)
    ax.legend(loc='upper right', frameon=False, fontsize=14)
    print(f'{tag}: Phi(E_min) best {x[0]/N:.3f}, band at E_min [{b[0,0]/N:.3f}, {b[0,1]/N:.3f}], '
          f'truth {Phi[np.searchsorted(Ef, 0.185)]/N:.3f};  first step at {edges[1+np.argmax(np.abs(np.diff(x)) > 1e-6*x[0])]:.3f} MeV')
axes[0].set_ylabel(r'$\Phi$ [$10^{12}$ cm$^{-2}$s$^{-1}$]', fontsize=22)
plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(RES, f'first_step_compare_1eV.{ext}'), dpi=115)
print('saved results/first_step_compare_1eV.pdf/.png')
