"""Compare the equalized best fits with those currently in Appendix C.

Appendix C now shows the spliced grid (published CRmat180 columns below 2 MeV, N_int = 272 / 286)
fitted to the original mock data Ratebin7 (no resolution). Its best fits are copied here as
data/appendixC_exactmerge_<thr>.npz (from ../spliced_crosscheck/results/exactmerge_<thr>.npz).
The equalized best fits (results/method1eq_bestfit_<thr>.npz) use the resolution mock data.

Printed for each threshold: number of downward steps, first-step energy, the two staircases and
the true flux at a few energies, and mean / max relative deviations below 2 MeV.

usage: python compare_with_appendixC.py
output: results/appC_vs_equalized.pdf/.png
"""
import os, sys, importlib
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CASES = (('1eV', 0.18, [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 1.9]),
         ('5eV', 0.41, [0.45, 0.5, 0.7, 1.0, 1.5, 1.9]))

def stair(z, E):
    return z['x'][np.clip(np.searchsorted(z['edges'], E, side='right') - 1, 0, len(z['x']) - 1)]

def steps(z):
    return int((np.abs(np.diff(z['x'])) > 1e-6*z['x'][0]).sum())

def first_step(z):
    return z['edges'][1 + np.argmax(np.abs(np.diff(z['x'])) > 1e-6*z['x'][0])]

def integrated(df, Ef):
    ph = np.interp(Ef, df['MeV'], df['cm**-2sec-1MeV-1'], right=0)
    return np.r_[np.cumsum((0.5*(ph[1:] + ph[:-1])*np.diff(Ef))[::-1])[::-1], 0.0]

plt.style.use(os.path.join(REPO, '1eV', 'physrev.mplstyle'))
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
for ax, (thr, emin, probes) in zip(axes, CASES):
    sys.path.insert(0, os.path.join(REPO, thr)); os.chdir(os.path.join(REPO, thr))
    import neutrino_analysis_band as nab; importlib.reload(nab)
    a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='scipy', T=3.0)
    os.chdir(HERE); sys.path.pop(0)
    Ef = np.linspace(0.1, 7.0, 690001)
    Phi = integrated(a.fig1Solid, Ef)
    C = np.load(os.path.join(HERE, 'data', f'appendixC_exactmerge_{thr}.npz'))
    Q = np.load(os.path.join(HERE, 'results', f'method1eq_bestfit_{thr}.npz'))

    print(f'=== {thr}: Appendix C (N_int {len(C["x"])}, Ratebin7)  vs  equalized (N_int {len(Q["x"])}, Ratebin7res)')
    print(f'   downward steps: {steps(C)} vs {steps(Q)};  first step at {first_step(C):.3f} vs {first_step(Q):.3f} MeV')
    print('   E [MeV]   App C   equalized   truth   (1e12 cm^-2 s^-1)')
    for E in probes:
        print(f'   {E:5.2f}    {stair(C, E)/1e12:6.3f}   {stair(Q, E)/1e12:6.3f}    {np.interp(E, Ef, Phi)/1e12:6.3f}')
    Eg = np.linspace(emin + 1e-6, 2.0, 20001); tr = np.interp(Eg, Ef, Phi)
    for name, z in (('App C', C), ('equalized', Q)):
        dev = np.abs(stair(z, Eg) - tr)/tr
        print(f'   {name:9s}: |fit - truth|/truth below 2 MeV: mean {dev.mean():.1%}, max {dev.max():.1%}')
    dd = np.abs(stair(C, Eg) - stair(Q, Eg))/tr
    print(f'   App C vs equalized below 2 MeV: mean {dd.mean():.1%}, max {dd.max():.1%} (at {Eg[np.argmax(dd)]:.3f} MeV)')

    ax.plot(Ef, Phi/1e12, color='black', lw=2, label='With NC')
    ax.plot(Ef, integrated(a.fig1dashed, Ef)/1e12, color='black', lw=2, ls='--', label='Without NC')
    ax.stairs(C['x']/1e12, C['edges'], baseline=None, color='#D55E00', lw=2.2, ls='--',
              label=f'Appendix C now ($N_{{\\rm int}}={len(C["x"])}$, original mock data)')
    ax.stairs(Q['x']/1e12, Q['edges'], baseline=None, color='#0072B2', lw=1.8,
              label=f'equalized ($N_{{\\rm int}}={len(Q["x"])}$, mock data with resolution)')
    ax.set_xscale('log'); ax.set_xlim(0.15, 3.0); ax.set_ylim(0.3, 2.1)
    ax.set_xlabel(r'$E_\nu$ [MeV]', fontsize=20); ax.tick_params(labelsize=14)
    ax.set_title(rf"$E'_{{\rm thr}} = {thr[0]}$ eV", fontsize=16)
    ax.legend(loc='lower left', frameon=False, fontsize=11)
axes[0].set_ylabel(r'$\Phi$ [$10^{12}$ cm$^{-2}$s$^{-1}$]', fontsize=20)
plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(HERE, 'results', f'appC_vs_equalized.{ext}'), dpi=115)
print('saved results/appC_vs_equalized.pdf/.png')
