"""N_int の比較図: 同一設定で 2 つの N_int を重ねる。
usage: python make_Nint_comparison.py <thr> <bkg> <N_a> <N_b>   例: 1eV none 80 160"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
plt.style.use('/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2/1eV/physrev.mplstyle')

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
thr, bkg = sys.argv[1], sys.argv[2]
NA, NB   = sys.argv[3], sys.argv[4]
LABEL = {'b': 'Exp Bkg', 'none': 'No Bkg', 'flat': 'Flat Bkg'}
tag  = f"{thr}_{'ExpBkg' if bkg=='b' else 'NoBkg' if bkg=='none' else bkg}"
name = f"{thr[0]} eV, {LABEL.get(bkg,bkg)}"
CA, CB = '#0072B2', '#D55E00'          # N_a, N_b

EMIN = {'1eV': 0.18, '5eV': 0.41}[thr]

def load(N):
    f = os.path.join(ROOT,'data',f'N{N}',f'bias_variance_{thr}_{bkg}_T3_M500_N{N}_vertex_on_bkgvar.json')
    d = json.load(open(f)); r = d['runs']['vertex_on']
    bf = np.array(d['best_fit_physical'])
    b  = np.array(r['bias']); sd = np.array(r['sd']); M = r['M_eff']
    n  = len(bf)
    Ev = np.linspace(EMIN, 2.0, n+1)[:-1]      # 区間の下端に打つ（N を倍にしても格子が入れ子になる）
    ok = bf > 0
    return Ev[ok], 100*b[ok]/bf[ok], 100*sd[ok]/bf[ok], 100*(sd[ok]/np.sqrt(M))/bf[ok]

EA, bA, sA, eA = load(NA)
EB, bB, sB, eB = load(NB)

def fin(ylab, logy=False, headroom=0.34, fs=15):
    ax = plt.gca()
    ax.set_xscale('log'); ax.set_xlim(0.1, 3)
    if logy:
        ax.set_yscale('log'); lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo*(hi/lo)**(1/(1-headroom)))
    else:
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo + (hi-lo)/(1-headroom))
    ax.set_xlabel(r"$E_\nu$ [MeV]", fontsize=30); ax.set_ylabel(ylab, fontsize=30)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0,10)*0.1, numticks=20))
    plt.tick_params(axis='both', which='major', labelsize=23)
    plt.tick_params(axis='both', which='minor', labelsize=23)
    plt.legend(loc='upper right', fontsize=fs, frameon=False, title=name, title_fontsize=fs)

# (1) relative bias
plt.figure(figsize=(8,6)); plt.axhline(0, color='k', lw=1.0)
plt.errorbar(EA, bA, yerr=eA, fmt='o', ms=5.0, mfc='none', mew=1.3, color=CA, ecolor=CA,
             elinewidth=1.0, capsize=2.4, capthick=0.9, zorder=2, label=rf'$N_{{\rm int}}={NA}$')
plt.errorbar(EB, bB, yerr=eB, fmt='s', ms=2.6, color=CB, ecolor=CB, elinewidth=0.7,
             capsize=1.6, capthick=0.6, zorder=3, label=rf'$N_{{\rm int}}={NB}$')
fin('relative bias [%]')
plt.savefig(os.path.join(RES, f'CMP_relbias_{tag}_N{NA}_vs_N{NB}.pdf'))
plt.savefig(os.path.join(RES, f'CMP_relbias_{tag}_N{NA}_vs_N{NB}.png'), dpi=115); plt.close()

# (2) relative dispersion
plt.figure(figsize=(8,6))
plt.plot(EA, sA, ls='none', marker='o', ms=5.0, mfc='none', mec=CA, mew=1.3, zorder=2, label=rf'$N_{{\rm int}}={NA}$')
plt.plot(EB, sB, ls='none', marker='s', ms=2.6, mfc=CB, mec=CB, zorder=3, label=rf'$N_{{\rm int}}={NB}$')
fin('relative dispersion [%]')
plt.savefig(os.path.join(RES, f'CMP_reldisp_{tag}_N{NA}_vs_N{NB}.pdf'))
plt.savefig(os.path.join(RES, f'CMP_reldisp_{tag}_N{NA}_vs_N{NB}.png'), dpi=115); plt.close()

# (3) b^2 / sigma^2
plt.figure(figsize=(8,6))
plt.axhline(1, color='k', lw=1.0, ls='--', label=r'ratio $=1$')
plt.plot(EA, (bA/sA)**2, ls='none', marker='o', ms=5.0, mfc='none', mec=CA, mew=1.3, zorder=2, label=rf'$N_{{\rm int}}={NA}$')
plt.plot(EB, (bB/sB)**2, ls='none', marker='s', ms=2.6, mfc=CB, mec=CB, zorder=3, label=rf'$N_{{\rm int}}={NB}$')
fin(r"$b_j^2/\sigma_j^2$", logy=True, headroom=0.28, fs=14)
plt.savefig(os.path.join(RES, f'CMP_bias2var_{tag}_N{NA}_vs_N{NB}.pdf'))
plt.savefig(os.path.join(RES, f'CMP_bias2var_{tag}_N{NA}_vs_N{NB}.png'), dpi=115); plt.close()

# (4) bias and dispersion together, both N_int
plt.figure(figsize=(8,6)); plt.axhline(0, color='k', lw=1.0)
plt.errorbar(EA, bA, yerr=eA, fmt='o', ms=5.0, mfc='none', mew=1.3, color=CA, ecolor=CA,
             elinewidth=1.0, capsize=2.4, capthick=0.9, zorder=2,
             label=rf'bias, $N_{{\rm int}}={NA}$')
plt.errorbar(EB, bB, yerr=eB, fmt='s', ms=2.6, color=CB, ecolor=CB, elinewidth=0.7,
             capsize=1.6, capthick=0.6, zorder=3, label=rf'bias, $N_{{\rm int}}={NB}$')
plt.plot(EA, sA, ls='none', marker='^', ms=5.0, mfc='none', mec='#009E73', mew=1.3, zorder=2,
         label=rf'dispersion, $N_{{\rm int}}={NA}$')
plt.plot(EB, sB, ls='none', marker='v', ms=3.0, mfc='#7B3294', mec='#7B3294', zorder=3,
         label=rf'dispersion, $N_{{\rm int}}={NB}$')
fin('relative value [%]', headroom=0.42, fs=13)
plt.savefig(os.path.join(RES, f'CMP_relbias_reldisp_{tag}_N{NA}_vs_N{NB}.pdf'))
plt.savefig(os.path.join(RES, f'CMP_relbias_reldisp_{tag}_N{NA}_vs_N{NB}.png'), dpi=115); plt.close()

print(f"[{name}]  N={NA}: {len(EA)} bins   N={NB}: {len(EB)} bins")
for lab, E, b_, s_ in ((NA,EA,bA,sA), (NB,EB,bB,sB)):
    r2 = (b_/s_)**2; mid = (E>0.5)&(E<1.3)
    print(f"  N={lab:>3}: median|relBias|={np.median(np.abs(b_)):5.2f}%  median relDisp={np.median(s_):5.1f}%  "
          f"median b2/s2={np.median(r2):.4f}  max={np.max(r2):.2f}  |  mid-range: {np.median(np.abs(b_[mid])):.2f}%, {np.median(r2[mid]):.4f}")
