"""1 eV の最初の段が真の曲線からずれて見える件の診断.

  1. ビンごとの閉包: 真の Phi を区間平均で離散化した x_true に対し M x_true / Ratebin7
     (curlyR 一様行列 = method1u, 旧行列 = method1 の両方)
  2. Delta chi2 = 0 帯 (method1u): E < 1 MeV の各区間で x_j の min / max (linprog)
  3. x_true が帯に入るか (= 真のフラックスがデータを厳密に再現する解か)

usage: python first_step_check.py      (1 eV)
出力: results/first_step_check_1eV.npz と図 results/first_step_check_1eV.pdf/.png
"""
import sys, os
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = '1eV'
RES = os.path.join(HERE, 'results')
ECUT = 1.0

sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='scipy', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
conv = a.cm**2*a.sec
data = np.asarray(a.Ratebin7, dtype=float)

Ef = np.linspace(0.18, 7.2, 400001)
phif = np.interp(Ef, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'], right=0)
cum = np.r_[np.cumsum((0.5*(phif[1:]+phif[:-1])*np.diff(Ef))[::-1])[::-1], 0.0]   # Phi(E)
Icum = np.r_[0.0, np.cumsum(0.5*(cum[1:]+cum[:-1])*np.diff(Ef))]                  # int_0^E Phi

def x_true(edges):                                   # 区間平均の Phi (物理単位)
    I = np.interp(edges, Ef, Icum)
    return np.diff(I)/np.diff(edges)

mats = {
    'method1u (curlyR, uniform)': (
        np.loadtxt(os.path.join(HERE, 'data', 'CRmat_method1u_1eV_originalUnit.csv'), delimiter=','),
        np.loadtxt(os.path.join(HERE, 'data', 'edges_method1u_1eV.csv'), delimiter=',')),
    'method1  (published+adaptive)': (
        np.loadtxt(os.path.join(HERE, 'data', 'CRmat_method1_1eV_originalUnit.csv'), delimiter=','),
        np.loadtxt(os.path.join(HERE, 'data', 'edges_method1_1eV.csv'), delimiter=',')),
}
bins = ['1-3', '3-5', '5-7', '7-9', '9-11']
print('closure M x_true / Ratebin7, first 5 bins and overall:')
for name, (M, e) in mats.items():
    r = (M*conv_mat) @ (x_true(e)/conv) / data
    print(f'  {name}: ' + '  '.join(f'{b} eV {v:.4f}' for b, v in zip(bins, r[:5]))
          + f'   | all bins {r.min():.4f}..{r.max():.4f}')

# ---------- Delta chi2 = 0 帯 (method1u) ----------
M, edges = mats['method1u (curlyR, uniform)']
n = M.shape[1]
XS = 1e12/conv
A = csr_matrix((M*conv_mat)*XS/data[:, None])
Dm = (eye(n, n, k=1) - eye(n, n)).tocsr()[:-1]

def lp(c):
    return linprog(c, A_ub=Dm, b_ub=np.zeros(n-1), A_eq=A, b_eq=np.ones(len(data)),
                   bounds=(0, None), method='highs', options={'time_limit': 60.0})

J = np.where(edges[:-1] < ECUT)[0]
band = np.zeros((len(J), 2))
for k, j in enumerate(J):
    c = np.zeros(n); c[j] = 1.0
    band[k] = (lp(c).fun*XS*conv, -lp(-c).fun*XS*conv)

xt = x_true(edges)
z = np.load(os.path.join(HERE, 'data', 'method1u_bestfit_1eV.npz'))
xb = z['x']
inside = (xt[J] >= band[:, 0]*(1-1e-6)) & (xt[J] <= band[:, 1]*(1+1e-6))
print(f'\nmethod1u: true flux inside the Delta chi2 = 0 band for {inside.sum()}/{len(J)} intervals below {ECUT} MeV')
print('   E [MeV]   best fit    band               truth   (1e12 cm^-2 s^-1)')
for E in [0.18, 0.2, 0.25, 0.3, 0.34, 0.36, 0.4, 0.45, 0.5]:
    k = np.searchsorted(edges[J], E, side='right')-1
    print(f'   {E:5.2f}    {xb[J[k]]/1e12:6.3f}    [{band[k,0]/1e12:6.3f}, {band[k,1]/1e12:6.3f}]   {xt[J[k]]/1e12:6.3f}')

np.savez(os.path.join(RES, 'first_step_check_1eV.npz'), edges=edges, J=J, band=band, x_true=xt, x_best=xb)

# ---------- 図 ----------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(os.path.join(REPO, '1eV', 'physrev.mplstyle'))
zo = np.load(os.path.join(RES, 'exactmerge_1eV.npz'))
N = 1e12
e_lo = edges[J[0]:J[-1]+2]
fig, ax = plt.subplots(figsize=(8, 6))
ax.stairs(band[:, 1]/N, e_lo, baseline=band[:, 0]/N, fill=True, color='#0072B2', alpha=0.18, lw=0,
          label=r'$\Delta\chi^2=0$ band (new $\mathcal{R}_{ij}$)')
ax.plot(Ef, cum/N, color='black', lw=2, label='true flux (with NC)')
ax.stairs(xb[J]/N, e_lo, baseline=None, color='#0072B2', lw=2.2, label=r'best fit, new $\mathcal{R}_{ij}$ ($N_{\rm int}=674$)')
eo = zo['edges']; jo = eo[:-1] < ECUT
ax.stairs(zo['x'][jo]/N, eo[:np.sum(jo)+1], baseline=None, color='#D55E00', lw=1.6, ls='--',
          label=r'best fit, old $\mathcal{R}_{ij}$ ($N_{\rm int}=272$)')
ax.set_xlim(0.15, ECUT); ax.set_ylim(0.9, 2.6)
ax.set_xlabel(r'$E_\nu$ [MeV]', fontsize=22); ax.set_ylabel(r'$\Phi$ [$10^{12}$ cm$^{-2}$s$^{-1}$]', fontsize=22)
ax.tick_params(labelsize=16)
ax.legend(loc='upper right', frameon=False, fontsize=13)
plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(RES, f'first_step_check_1eV.{ext}'), dpi=115)
print('saved results/first_step_check_1eV.pdf/.png')
