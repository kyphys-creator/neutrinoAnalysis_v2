"""同一 N_int で M=500 と M=10000 を重ねる比較図（bias と dispersion のみ）。
usage: make_M_comparison.py <thr> <bkg> <N_int> <M_a> <M_b>   (M_b が上に描画される)"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
plt.style.use('/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2/1eV/physrev.mplstyle')

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
thr, bkg, NINT, MA, MB = sys.argv[1:6]
RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
name = f"{thr[0]} eV, {'Exp Bkg' if bkg=='b' else 'No Bkg'}"
tag  = f"{thr}_{'ExpBkg' if bkg=='b' else 'NoBkg'}"
CA, CB = '#0072B2', '#D55E00'   # A=M500 blue open, B=M10000 orange filled (on top)

def load(M):
    d = json.load(open(os.path.join(ROOT,'data',f'N{NINT}',
        f'bias_variance_{thr}_{bkg}_T3_M{M}_N{NINT}_vertex_on_bkgvar.json')))
    r = d['runs']['vertex_on']
    Ev = np.array(d['Ev_MeV']); bf = np.array(d['best_fit_physical'])
    b = np.array(r['bias']); sd = np.array(r['sd']); N = r['M_eff']
    ok = bf > 0
    return Ev[ok], 100*b[ok]/bf[ok], 100*sd[ok]/bf[ok], 100*(sd[ok]/np.sqrt(N))/bf[ok]

EA,bA,sA,eA = load(MA); EB,bB,sB,eB = load(MB)

def fin(ylab, headroom=0.40):
    plt.xscale('log'); plt.xlim(0.1,3)
    ax=plt.gca(); ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.,10)*0.1, numticks=20))
    plt.tick_params(axis='both', which='both', labelsize=17)
    plt.xlabel(r"$E_\nu$ [MeV]", fontsize=24); plt.ylabel(ylab, fontsize=23)
    lo,hi=plt.ylim(); plt.ylim(lo, lo+(hi-lo)/(1-headroom))
    plt.legend(loc='upper right', fontsize=15, frameon=False, title=name, title_fontsize=15)
    plt.tight_layout()

# (1) relative bias
plt.figure(figsize=(8,6)); plt.axhline(0, color='k', lw=1.0)
plt.errorbar(EA, bA, yerr=eA, fmt='o', ms=5.0, mfc='none', mew=1.3, color=CA, ecolor=CA,
             elinewidth=1.0, capsize=2.4, capthick=0.9, zorder=2, label=rf'$M={MA}$')
plt.errorbar(EB, bB, yerr=eB, fmt='s', ms=2.6, color=CB, ecolor=CB, elinewidth=0.7,
             capsize=1.6, capthick=0.6, zorder=3, label=rf'$M=10^4$' if MB=='10000' else rf'$M={MB}$')
fin('relative bias [%]')
plt.savefig(os.path.join(RES, f'CMPM_relbias_{tag}_N{NINT}_M{MA}_vs_M{MB}.pdf'))
plt.savefig(os.path.join(RES, f'CMPM_relbias_{tag}_N{NINT}_M{MA}_vs_M{MB}.png'), dpi=115); plt.close()

# (2) relative dispersion
plt.figure(figsize=(8,6))
plt.plot(EA, sA, ls='none', marker='o', ms=5.0, mfc='none', mec=CA, mew=1.3, zorder=2, label=rf'$M={MA}$')
plt.plot(EB, sB, ls='none', marker='s', ms=2.6, mfc=CB, mec=CB, zorder=3,
         label=rf'$M=10^4$' if MB=='10000' else rf'$M={MB}$')
fin('relative dispersion [%]', headroom=0.30)
plt.savefig(os.path.join(RES, f'CMPM_reldisp_{tag}_N{NINT}_M{MA}_vs_M{MB}.pdf'))
plt.savefig(os.path.join(RES, f'CMPM_reldisp_{tag}_N{NINT}_M{MA}_vs_M{MB}.png'), dpi=115); plt.close()

db = bB-bA; ds = sB-sA
print(f"[{name}] N={NINT}  M={MA} -> M={MB}")
print(f"  relBias : median shift {np.median(db):+.2f} pt  max|shift| {np.max(np.abs(db)):.2f} pt")
print(f"  relDisp : median shift {np.median(ds):+.2f} pt  max|shift| {np.max(np.abs(ds)):.2f} pt")
print(f"  MC err on relBias (M={MB}): median {np.median(eB):.2f}%")
