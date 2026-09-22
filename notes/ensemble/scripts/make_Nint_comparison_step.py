"""N_int 比較（ステップ表示）: delta Phi_j は区間上で定数なので、実際の区間幅にわたる
階段関数として描く。N が倍なら細いステップが太いステップに入れ子になり、中点のずれが生じない。
usage: python make_Nint_comparison_step.py <thr> <bkg> <N_a> <N_b>"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
plt.style.use('/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2/1eV/physrev.mplstyle')

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
thr, bkg, NA, NB = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
EMIN = {'1eV': 0.18, '5eV': 0.41}[thr]
LABEL = {'b':'Exp Bkg','none':'No Bkg','flat':'Flat Bkg'}
tag  = f"{thr}_{'ExpBkg' if bkg=='b' else 'NoBkg' if bkg=='none' else bkg}"
name = f"{thr[0]} eV, {LABEL.get(bkg,bkg)}"
CA, CB = '#0072B2', '#D55E00'

def load(N):
    f = os.path.join(ROOT,'data',f'N{N}',f'bias_variance_{thr}_{bkg}_T3_M500_N{N}_vertex_on_bkgvar.json')
    d = json.load(open(f)); r = d['runs']['vertex_on']
    bf = np.array(d['best_fit_physical']); b = np.array(r['bias']); sd = np.array(r['sd'])
    n = len(bf); edges = np.linspace(EMIN, 2.0, n+1)
    ok = bf > 0
    relb = np.where(ok, 100*b/np.where(ok,bf,1), np.nan)
    rels = np.where(ok, 100*sd/np.where(ok,bf,1), np.nan)
    return edges, relb, rels

def stepxy(edges, y):
    """区間ごとに一定の階段。NaN の区間は線を切る。"""
    x = np.repeat(edges, 2)[1:-1]
    return x, np.repeat(y, 2)

def fin(ylab, logy=False, headroom=0.34, fs=15):
    ax = plt.gca(); ax.set_xscale('log'); ax.set_xlim(0.1, 3)
    if logy:
        ax.set_yscale('log'); lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo*(hi/lo)**(1/(1-headroom)))
    else:
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo + (hi-lo)/(1-headroom))
    ax.set_xlabel(r"$E_\nu$ [MeV]", fontsize=30); ax.set_ylabel(ylab, fontsize=30)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0,10)*0.1, numticks=20))
    plt.tick_params(axis='both', which='major', labelsize=23)
    plt.tick_params(axis='both', which='minor', labelsize=23)
    plt.legend(loc='upper right', fontsize=fs, frameon=False, title=name, title_fontsize=fs)

eA, bA, sA = load(NA)
eB, bB, sB = load(NB)

for what, yA, yB, ylab, fname, logy in (
        ('relbias', bA, bB, 'relative bias [%]',       f'CMPstep_relbias_{tag}_N{NA}_vs_N{NB}', False),
        ('reldisp', sA, sB, 'relative dispersion [%]', f'CMPstep_reldisp_{tag}_N{NA}_vs_N{NB}', False)):
    plt.figure(figsize=(8,6))
    if not logy: plt.axhline(0, color='k', lw=1.0)
    xB, yBs = stepxy(eB, yB); xA, yAs = stepxy(eA, yA)
    plt.plot(xB, yBs, color=CB, ls='-', lw=1.1, alpha=0.85, label=rf'$N_{{\rm int}}={NB}$')
    plt.plot(xA, yAs, color=CA, ls='-', lw=2.2,            label=rf'$N_{{\rm int}}={NA}$')
    fin(ylab, logy=logy)
    plt.savefig(os.path.join(RES, fname+'.pdf')); plt.savefig(os.path.join(RES, fname+'.png'), dpi=115)
    plt.close()

# b^2/sigma^2
plt.figure(figsize=(8,6))
plt.axhline(1, color='k', lw=1.0, ls='--', label=r'ratio $=1$')
xB, r2B = stepxy(eB, (bB/sB)**2); xA, r2A = stepxy(eA, (bA/sA)**2)
plt.plot(xB, r2B, color=CB, ls='-', lw=1.1, alpha=0.85, label=rf'$N_{{\rm int}}={NB}$')
plt.plot(xA, r2A, color=CA, ls='-', lw=2.2,            label=rf'$N_{{\rm int}}={NA}$')
fin(r"$b_j^2/\sigma_j^2$", logy=True, headroom=0.28, fs=14)
plt.savefig(os.path.join(RES, f'CMPstep_bias2var_{tag}_N{NA}_vs_N{NB}.pdf'))
plt.savefig(os.path.join(RES, f'CMPstep_bias2var_{tag}_N{NA}_vs_N{NB}.png'), dpi=115); plt.close()
print(f"saved CMPstep_* for {name}: N={NA} vs N={NB}")
