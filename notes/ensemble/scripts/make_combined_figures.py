"""4 設定を 2x2 で並べた比較図。Graciela の「他の背景でも計算して比較せよ」への直接の答え。"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
plt.style.use('/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2/1eV/physrev.mplstyle')

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
NINT = sys.argv[1] if len(sys.argv) > 1 else '80'
M    = sys.argv[2] if len(sys.argv) > 2 else '500'
SUF  = f'N{NINT}' + (f'_M{M}' if M != '500' else '')
DATA = os.path.join(ROOT, 'data', f'N{NINT}'); RES = os.path.join(ROOT, 'results')
BL, VE = '#0072B2', '#D55E00'
CONFIGS = [('1eV','b','1 eV, Exp Bkg'), ('1eV','none','1 eV, No Bkg'),
           ('5eV','b','5 eV, Exp Bkg'), ('5eV','none','5 eV, No Bkg')]

def load(thr,bkg):
    d = json.load(open(os.path.join(DATA, f'bias_variance_{thr}_{bkg}_T3_M{M}_N{NINT}_vertex_on_bkgvar.json')))
    r = d['runs']['vertex_on']
    Ev = np.array(d['Ev_MeV']); bf = np.array(d['best_fit_physical'])
    b = np.array(r['bias']); sd = np.array(r['sd']); N = r['M_eff']
    ok = bf > 0
    return Ev[ok], b[ok]/bf[ok], sd[ok]/bf[ok], (sd[ok]/np.sqrt(N))/bf[ok]

def axfmt(ax, xlab, ylab, logy=False):
    ax.set_xscale('log'); ax.set_xlim(0.1, 3)
    if logy: ax.set_yscale('log')
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0,10)*0.1, numticks=20))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    if xlab: ax.set_xlabel(r"$E_\nu$ [MeV]", fontsize=20)
    if ylab: ax.set_ylabel(ylab, fontsize=19)

# ---- (1) relative bias & relative dispersion, 2x2 ----
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
for ax, (thr,bkg,name) in zip(axes.ravel(), CONFIGS):
    E, rb, rs, eb = load(thr,bkg)
    ax.axhline(0, color='k', lw=1.0)
    ax.errorbar(E, 100*rb, yerr=100*eb, fmt='o', ms=3.2, color=BL, ecolor=BL,
                elinewidth=0.9, capsize=2.2, capthick=0.8,
                label=r'relative bias  $b_j/\delta\Phi_j^{\rm Best\ fit}$')
    ax.plot(E, 100*rs, ls='none', marker='s', ms=3.2, mfc=VE, mec=VE,
            label=r'relative dispersion  $\sigma_j/\delta\Phi_j^{\rm Best\ fit}$')
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo + (hi-lo)/(1-0.30))
    ax.set_title(name, fontsize=19)
    axfmt(ax, True, 'relative value [%]')
axes[0,0].legend(loc='upper right', fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES,f'ALL_relbias_reldisp_{SUF}.pdf'))
plt.savefig(os.path.join(RES,f'ALL_relbias_reldisp_{SUF}.png'), dpi=115); plt.close()

# ---- (2) b^2 / sigma^2, 2x2 ----
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
for ax, (thr,bkg,name) in zip(axes.ravel(), CONFIGS):
    E, rb, rs, _ = load(thr,bkg)
    r2 = (rb/rs)**2
    ax.axhline(1, color='k', lw=1.0, ls='--', label=r'ratio $=1$ (optimized Tikhonov)')
    ax.plot(E, r2, ls='none', marker='o', ms=3.2, mfc=BL, mec=BL, label=r'$b_j^2/\sigma_j^2$')
    ax.set_yscale('log')
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo*(hi/lo)**(1/(1-0.26)))
    ax.set_title(name, fontsize=19)
    axfmt(ax, True, r"$b_j^2/\sigma_j^2$", logy=True)
axes[0,0].legend(loc='upper right', fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES,f'ALL_bias2_over_var_{SUF}.pdf'))
plt.savefig(os.path.join(RES,f'ALL_bias2_over_var_{SUF}.png'), dpi=115); plt.close()
print(f"saved ALL_relbias_reldisp_{SUF} and ALL_bias2_over_var_{SUF}")

# summary numbers
print(f"\n{'config':16s}{'median |relBias|':>18}{'median relDisp':>16}{'median b2/s2':>14}{'max b2/s2':>11}")
for thr,bkg,name in CONFIGS:
    E, rb, rs, _ = load(thr,bkg); r2=(rb/rs)**2
    print(f"{name:16s}{np.median(np.abs(rb)):18.3f}{np.median(rs):16.3f}{np.median(r2):14.4f}{np.max(r2):11.3f}")
