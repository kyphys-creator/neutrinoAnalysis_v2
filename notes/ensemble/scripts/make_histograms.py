"""Bob Cousins が要求したヒストグラム: point estimate と (point estimate - true) の分布。
真値は入力フラックス (Mathematica/<thr>/input/dNdEsolid.csv) から計算した区間平均。
4 設定すべてについて results/<tag>/ に出力する。"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
plt.style.use('/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2/1eV/physrev.mplstyle')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
DATA = os.path.join(ROOT, 'data', 'N80')
RES  = os.path.join(ROOT, 'results')
BL, VE, GR = '#0072B2', '#D55E00', '#009E73'
EMIN = {'1eV': 0.18, '5eV': 0.41}
CONFIGS = [('1eV','b','1eV_ExpBkg'), ('1eV','none','1eV_NoBkg'),
           ('5eV','b','5eV_ExpBkg'), ('5eV','none','5eV_NoBkg')]

def true_dphi(thr, n):
    """区間平均した真の deltaPhi_j [cm^-2 s^-1]"""
    norm = (2.65e22/205.3)/(4*np.pi)*(1/7200**2 + 1/10200**2)
    f = pd.read_csv(os.path.join(REPO, 'Mathematica', thr, 'input', 'dNdEsolid.csv'), header=None)
    E, ph = f[0].values, f[1].values*norm
    Eg = np.linspace(E.min(), E.max(), 40001); pg = np.interp(Eg, E, ph)
    c = cumulative_trapezoid(pg, Eg, initial=0)
    P = lambda x: np.interp(x, Eg, c[-1]-c)
    edges = np.linspace(EMIN[thr], 2.0, n+1)
    return np.array([np.trapezoid(P(np.linspace(a,b,201))-P(2.0), np.linspace(a,b,201))/(b-a)
                     for a, b in zip(edges[:-1], edges[1:])])

TRUE = {thr: None for thr in EMIN}
for thr, bkg, tag in CONFIGS:
    d = json.load(open(os.path.join(DATA, f'bias_variance_{thr}_{bkg}_T3_M500_N80_vertex_on_bkgvar.json')))
    r = d['runs']['vertex_on']
    X  = np.array(r['samples'])/1e12
    bf = np.array(d['best_fit_physical'])/1e12
    Ev = np.array(d['Ev_MeV']); n = len(Ev)
    if TRUE[thr] is None: TRUE[thr] = true_dphi(thr, n)/1e12
    tr = TRUE[thr]
    outdir = os.path.join(RES, tag); os.makedirs(outdir, exist_ok=True)
    sel = [0, n//10, n//4, n//2, int(0.875*n)]

    # (1) point estimates
    fig, axes = plt.subplots(1, len(sel), figsize=(19, 4.2))
    for ax, j in zip(axes, sel):
        ax.hist(X[:,j], bins=32, color=BL, alpha=.55, edgecolor='none')
        ax.axvline(tr[j], color='k',  lw=2.0,           label=r'true $\delta\Phi_j$ (input flux)')
        ax.axvline(bf[j], color=GR,   lw=2.0, ls='--',  label=r'$\delta\Phi_j^{\rm Best\ fit}$')
        ax.axvline(X[:,j].mean(), color=VE, lw=2.0, ls='-.', label=r'$\langle\delta\hat\Phi_j\rangle$')
        ax.set_title(rf'$E_\nu={Ev[j]:.2f}$ MeV', fontsize=17)
        ax.set_xlabel(r'$\delta\hat\Phi_j\ [10^{12}\,{\rm cm^{-2}s^{-1}}]$', fontsize=15)
        ax.tick_params(axis='both', which='both', labelsize=13)
    axes[0].set_ylabel('pseudo-experiments', fontsize=15)
    axes[0].legend(fontsize=12, frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'hist_deltaPhi_{tag}_N80.pdf'))
    plt.savefig(os.path.join(outdir, f'hist_deltaPhi_{tag}_N80.png'), dpi=110); plt.close()

    # (2) residuals w.r.t. the true input flux
    fig, axes = plt.subplots(1, len(sel), figsize=(19, 4.2))
    for ax, j in zip(axes, sel):
        res = X[:,j] - tr[j]
        ax.hist(res, bins=32, color=VE, alpha=.55, edgecolor='none')
        ax.axvline(0, color='k', lw=2.0, label='zero (unbiased)')
        ax.axvline(res.mean(), color=BL, lw=2.0, ls='-.', label=r'bias $b_j$')
        ax.set_title(rf'$E_\nu={Ev[j]:.2f}$ MeV', fontsize=17)
        ax.set_xlabel(r'$\delta\hat\Phi_j-\delta\Phi_j^{\rm true}\ [10^{12}\,{\rm cm^{-2}s^{-1}}]$', fontsize=14)
        ax.tick_params(axis='both', which='both', labelsize=13)
    axes[0].set_ylabel('pseudo-experiments', fontsize=15)
    axes[0].legend(fontsize=12, frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'hist_residual_{tag}_N80.pdf'))
    plt.savefig(os.path.join(outdir, f'hist_residual_{tag}_N80.png'), dpi=110); plt.close()

    bt = X.mean(0) - tr
    print(f"[{tag}] bins at j={sel}  RMS bias vs input flux = {np.sqrt((bt**2).mean()):.4f}e12"
          f"  |  vs best fit = {np.sqrt(((X.mean(0)-bf)**2).mean()):.4f}e12")
