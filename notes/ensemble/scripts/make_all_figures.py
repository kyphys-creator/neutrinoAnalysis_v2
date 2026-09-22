import json, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
plt.style.use('/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2/1eV/physrev.mplstyle')
BL,VE='#0072B2','#D55E00'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # notes/ensemble
DATA = os.path.join(ROOT, 'data', 'N80')
RES  = os.path.join(ROOT, 'results')

CONFIGS = [
    ('1eV','b','1eV_ExpBkg'),
    ('1eV','none','1eV_NoBkg'),
    ('5eV','none','5eV_NoBkg'),
    ('5eV','b','5eV_ExpBkg'),
]

def frame(xlabel, ylabel, legend_fs=17, loc='upper right', headroom=0.40, logy=False):
    """headroom = fraction of the final axis height left empty at the top for the legend."""
    ax = plt.gca()
    plt.xscale('log'); plt.xlim(0.1,3)
    if logy:
        plt.yscale('log')
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, lo*(hi/lo)**(1/(1-headroom)))
    else:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, lo + (hi-lo)/(1-headroom))
    plt.xlabel(xlabel, fontsize=30); plt.ylabel(ylabel, fontsize=30)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0,10)*0.1, numticks=20))
    plt.tick_params(axis='both', which='major', labelsize=23)
    plt.tick_params(axis='both', which='minor', labelsize=23)
    plt.legend(loc=loc, fontsize=legend_fs, frameon=False)

for thr, bkg, tag in CONFIGS:
    fn = os.path.join(DATA, f'bias_variance_{thr}_{bkg}_T3_M500_N80_vertex_on_bkgvar.json')
    d = json.load(open(fn))
    outdir = os.path.join(RES, tag); os.makedirs(outdir, exist_ok=True)
    r = d['runs']['vertex_on']
    Ev = np.array(d['Ev_MeV']); bf = np.array(d['best_fit_physical'])
    b = np.array(r['bias']); sd = np.array(r['sd']); N = r['M_eff']

    # ---- 1. bias (physical units), errorbar + fill_between, same style as before ----
    plt.figure(figsize=(8,6)); plt.axhline(0,color='k',lw=1.0)
    plt.fill_between(Ev, (b-sd)/1e12, (b+sd)/1e12, color=BL, alpha=0.15, lw=0,
                      label=r'$b_j\pm\sigma_j$ (dispersion of the bootstrap best fit $\delta\hat{\Phi}_j$)')
    plt.errorbar(Ev, b/1e12, yerr=sd/1e12, fmt='o', ms=3.5, color=BL, ecolor=BL,
                 elinewidth=0.9, capsize=2.5, capthick=0.9, alpha=0.9, zorder=3,
                 label=r'bias  $b_j=\langle\delta\hat\Phi_j\rangle-\delta\Phi_j^{\rm Best\ fit}$')
    frame(r"$E_\nu$ [MeV]", r"bias $[10^{12}\ \mathrm{cm^{-2}\,s^{-1}}]$")
    plt.savefig(os.path.join(outdir, f'bias_{tag}_N80_vertexon.pdf'))
    plt.savefig(os.path.join(outdir, f'bias_{tag}_N80_vertexon.png'), dpi=115)
    plt.close()

    # ---- relative quantities (bf as denominator; drop bins where bf == 0) ----
    ok = bf > 0
    E, B, S, BF = Ev[ok], b[ok], sd[ok], bf[ok]
    relB, relS = B/BF, S/BF
    errB = (S/np.sqrt(N))/BF
    r2 = (B**2)/(S**2)
    nexcl = int((~ok).sum())

    # ---- 2. relative bias & relative dispersion together ----
    plt.figure(figsize=(8,6)); plt.axhline(0,color='k',lw=1.0)
    plt.errorbar(E, 100*relB, yerr=100*errB, fmt='o', ms=3.5, color=BL, ecolor=BL,
                 elinewidth=0.9, capsize=2.2, capthick=0.8,
                 label=r'relative bias  $b_j/\delta\Phi_j^{\rm Best\ fit}$')
    plt.plot(E, 100*relS, ls='none', marker='s', ms=3.5, mfc=VE, mec=VE,
              label=r'relative dispersion  $\sigma_j/\delta\Phi_j^{\rm Best\ fit}$')
    frame(r"$E_\nu$ [MeV]", 'relative value [%]')
    plt.savefig(os.path.join(outdir, f'relbias_reldisp_{tag}_N80.pdf'))
    plt.savefig(os.path.join(outdir, f'relbias_reldisp_{tag}_N80.png'), dpi=115)
    plt.close()

    # ---- 3. squares ----
    plt.figure(figsize=(8,6))
    plt.plot(E, relB**2, ls='none', marker='o', ms=3.5, mfc=BL, mec=BL,
              label=r'(relative bias)$^2$  $(b_j/\delta\Phi_j^{\rm Best\ fit})^2$')
    plt.plot(E, relS**2, ls='none', marker='s', ms=3.5, mfc=VE, mec=VE,
              label=r'(relative dispersion)$^2$  $(\sigma_j/\delta\Phi_j^{\rm Best\ fit})^2$')
    frame(r"$E_\nu$ [MeV]", r"relative value squared", legend_fs=15, headroom=0.38, logy=True)
    plt.savefig(os.path.join(outdir, f'relbias2_reldisp2_{tag}_N80.pdf'))
    plt.savefig(os.path.join(outdir, f'relbias2_reldisp2_{tag}_N80.png'), dpi=115)
    plt.close()

    # ---- 4. b^2/sigma^2 ----
    plt.figure(figsize=(8,6))
    plt.axhline(1, color='k', lw=1.0, ls='--', label=r'ratio $=1$ (optimized Tikhonov)')
    plt.plot(E, r2, ls='none', marker='o', ms=3.5, mfc=BL, mec=BL, label=r'$b_j^2/\sigma_j^2$ (our method)')
    frame(r"$E_\nu$ [MeV]", r"$b_j^2/\sigma_j^2$", legend_fs=16, headroom=0.32, logy=True)
    plt.savefig(os.path.join(outdir, f'bias2_over_var_{tag}_N80.pdf'))
    plt.savefig(os.path.join(outdir, f'bias2_over_var_{tag}_N80.png'), dpi=115)
    plt.close()

    print(f"[{tag}] N={N}  d={d['d_bins']}  n_int={d['n_int']}  excluded {nexcl} bins (best fit = 0) from relative plots")
    for j in [0,2,8,20,45,70]:
        if j >= len(Ev): continue
        print(f"   j={j:2d} Ev={Ev[j]:.2f}  bias={b[j]/1e12:+.3f}e12  sd={sd[j]/1e12:.3f}e12"
              + (f"  relBias={b[j]/bf[j]:+.1%}  relDisp={sd[j]/bf[j]:.1%}  b2/s2={(b[j]/sd[j])**2:.3f}" if bf[j]>0 else "  (bf=0, excluded from relative plots)"))
