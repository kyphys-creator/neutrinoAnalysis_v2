"""Graciela の指示した論文用 3 図 (2026-09-05 のメール + 手書きノート pp.2-3):
  (1) 相対 b_j, sigma_j, sqrt(MSE_j)  [best fit で規格化, %]
  (2) b_j^2, sigma_j^2, MSE_j  [絶対値, flux^2 の単位]
  (3) b_j^2 / MSE_j
MSE_j = b_j^2 + sigma_j^2.  データは N_int=160, M=10^4, 背景変動あり。
usage: make_mse_figures.py [N_int] [M]
"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
plt.style.use('/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2/1eV/physrev.mplstyle')

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
NINT = sys.argv[1] if len(sys.argv) > 1 else '160'
M    = sys.argv[2] if len(sys.argv) > 2 else '10000'
THR  = sys.argv[3] if len(sys.argv) > 3 else None   # e.g. '1eV' -> only that threshold, 1x2
if THR == 'all': THR = None                          # 2x2 のまま TRIM だけ使う用
TRIM = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 高エネルギー端の点を落とす数
COMMON = 'common' in sys.argv[5:]   # 全パネル共通の縦軸 + 参照線 (Graciela 2026-09-11)
DATA = os.path.join(ROOT, 'data', f'N{NINT}')
RES  = os.path.join(ROOT, 'results', 'paper'); os.makedirs(RES, exist_ok=True)
SUF  = (f'{THR}_' if THR else '') + f'N{NINT}_M{M}' + ('_nolast' if TRIM else '') + ('_common' if COMMON else '')
CB, CO, CG = '#0072B2', '#D55E00', '#009E73'
CONFIGS = [('1eV','none','1 eV, No Bkg'), ('1eV','b','1 eV, Exp Bkg'),
           ('5eV','none','5 eV, No Bkg'), ('5eV','b','5 eV, Exp Bkg')]
if THR:
    CONFIGS = [c for c in CONFIGS if c[0] == THR]
GRID = (2,2) if len(CONFIGS) == 4 else (1,len(CONFIGS))
FSIZE = (13.5, 9.5) if len(CONFIGS) == 4 else (13.5, 5.2)

def grid_axes():
    fig, axes = plt.subplots(*GRID, figsize=FSIZE)
    import numpy as _np
    return fig, _np.atleast_1d(axes).ravel()

def load(thr, bkg):
    d = json.load(open(os.path.join(DATA, f'bias_variance_{thr}_{bkg}_T3_M{M}_N{NINT}_vertex_on_bkgvar.json')))
    r = d['runs']['vertex_on']
    Ev = np.array(d['Ev_MeV']); bf = np.array(d['best_fit_physical'])
    b  = np.array(r['bias']);   sd = np.array(r['sd']); Meff = r['M_eff']
    return Ev, bf, b, sd, Meff

def axfmt(ax, ylab, logy=False):
    ax.set_xscale('log'); ax.set_xlim(0.1, 3)
    if logy: ax.set_yscale('log')
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0,10)*0.1, numticks=20))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_xlabel(r"$E_\nu$ [MeV]", fontsize=20)
    ax.set_ylabel(ylab, fontsize=19)

def headroom(ax, frac, logy=False):
    lo, hi = ax.get_ylim()
    if logy: ax.set_ylim(lo, lo*(hi/lo)**(1/(1-frac)))
    else:    ax.set_ylim(lo, lo + (hi-lo)/(1-frac))

# ---- (1) relative b, sigma, sqrt(MSE) ----
fig, axes = grid_axes()
for ax, (thr,bkg,name) in zip(axes, CONFIGS):
    Ev, bf, b, sd, Meff = load(thr,bkg)
    ok = np.flatnonzero(bf > 0)
    if TRIM: ok = ok[:-TRIM]
    E, rb, rs = Ev[ok], 100*b[ok]/bf[ok], 100*sd[ok]/bf[ok]
    rmse = 100*np.sqrt(b[ok]**2 + sd[ok]**2)/bf[ok]
    ax.axhline(0, color='k', lw=1.0)
    ax.plot(E, rb, ls='none', marker='o', ms=3.2, mfc=CB, mec=CB, zorder=2,
            label=r'relative bias  $b_j/\delta\Phi_j^{\rm Best\ fit}$')
    ax.plot(E, rs, ls='none', marker='s', ms=3.2, mfc=CO, mec=CO, zorder=3,
            label=r'relative dispersion  $\sigma_j/\delta\Phi_j^{\rm Best\ fit}$')
    ax.plot(E, rmse, ls='none', marker='^', ms=4.6, mfc='none', mec=CG, mew=1.2, zorder=4,
            label=r'$\sqrt{{\rm MSE}_j}\,/\,\delta\Phi_j^{\rm Best\ fit}$')
    if COMMON:
        for y in (10, -10): ax.axhline(y, color='0.55', lw=0.8, ls='--', zorder=1)
    ax.set_title(name, fontsize=19)
    axfmt(ax, 'relative value [%]')
    if COMMON: ax.set_ylim(-70, 210)
    else: headroom(ax, 0.34)
axes[0].legend(loc='upper right', fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'relative_bias_disp_rmse_{SUF}.pdf'))
plt.savefig(os.path.join(RES, f'relative_bias_disp_rmse_{SUF}.png'), dpi=115); plt.close()

# ---- (2) b^2, sigma^2, MSE (absolute, flux^2 units) ----
fig, axes = grid_axes()
for ax, (thr,bkg,name) in zip(axes, CONFIGS):
    Ev, bf, b, sd, Meff = load(thr,bkg)
    if TRIM: Ev, bf, b, sd = Ev[:-TRIM], bf[:-TRIM], b[:-TRIM], sd[:-TRIM]
    mse = b**2 + sd**2
    pos = b**2 > 0
    ax.plot(Ev, sd**2, ls='none', marker='s', ms=3.2, mfc=CO, mec=CO, zorder=3,
            label=r'$\sigma_j^2$')
    ax.plot(Ev[pos], (b**2)[pos], ls='none', marker='o', ms=3.2, mfc=CB, mec=CB, zorder=2,
            label=r'$b_j^2$')
    ax.plot(Ev, mse, ls='none', marker='^', ms=4.6, mfc='none', mec=CG, mew=1.2, zorder=4,
            label=r'${\rm MSE}_j = b_j^2+\sigma_j^2$')
    ax.set_title(name, fontsize=19)
    axfmt(ax, r'$[{\rm cm^{-4}\,s^{-2}}]$', logy=True)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=99))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10), numticks=999))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if COMMON: ax.set_ylim(1e14, 1e25)
    else: headroom(ax, 0.14, logy=True)
axes[0].legend(loc='lower left' if COMMON else 'upper right', fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'abs_b2_s2_mse_{SUF}.pdf'))
plt.savefig(os.path.join(RES, f'abs_b2_s2_mse_{SUF}.png'), dpi=115); plt.close()

# ---- (3) b^2 / MSE ----
fig, axes = grid_axes()
for ax, (thr,bkg,name) in zip(axes, CONFIGS):
    Ev, bf, b, sd, Meff = load(thr,bkg)
    if TRIM: Ev, bf, b, sd = Ev[:-TRIM], bf[:-TRIM], b[:-TRIM], sd[:-TRIM]
    r = b**2/(b**2 + sd**2)
    ax.axhline(0.5, color='k', lw=1.0, ls='--', label=r'$b_j=\sigma_j$')
    if COMMON:
        for y in (0.1, 1.0): ax.axhline(y, color='0.55', lw=0.8, ls='--', zorder=1)
    ax.plot(Ev, r, ls='none', marker='o', ms=3.2, mfc=CB, mec=CB,
            label=r'$b_j^2/{\rm MSE}_j$')
    ax.set_yscale('log')
    ax.set_ylim((1e-8, 2) if COMMON else (None, 1.5))
    ax.set_title(name, fontsize=19)
    axfmt(ax, r'$b_j^2/{\rm MSE}_j$', logy=True)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=99))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10), numticks=999))
    ax.yaxis.set_minor_formatter(NullFormatter())
axes[0].legend(loc='lower left', fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'ratio_b2_over_mse_{SUF}.pdf'))
plt.savefig(os.path.join(RES, f'ratio_b2_over_mse_{SUF}.png'), dpi=115); plt.close()

print(f"saved 3 figures in {RES} (suffix {SUF})")
for thr,bkg,name in CONFIGS:
    Ev, bf, b, sd, Meff = load(thr,bkg)
    ok = bf > 0; r = b**2/(b**2+sd**2)
    rmse = np.sqrt(b[ok]**2+sd[ok]**2)/bf[ok]
    print(f"  {name:14s} median rel sqrtMSE={100*np.median(rmse):6.1f}%   "
          f"median b2/MSE={np.median(r):.4f}   max b2/MSE={r.max():.3f}")
