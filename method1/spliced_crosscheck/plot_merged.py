"""Figure of the merged best fit (band_comparison style). usage: plot_merged.py <thr>"""
import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = sys.argv[1] if len(sys.argv)>1 else '1eV'
RES=os.path.join(HERE,'results')
z = np.load(os.path.join(RES, f'merged_{THR}.npz'))
edges, xm, xo = z['edges'], z['x_merged'], z['x_orig']

sys.path.insert(0, os.path.join(REPO,THR)); os.chdir(os.path.join(REPO,THR))
import neutrino_analysis_band as nab
import pandas as pd
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='osqp', T=3.0)
xg = np.logspace(-2, np.log10(7.0), 4000)
ys = np.interp(xg, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'])
yd = np.interp(xg, a.fig1dashed['MeV'], a.fig1dashed['cm**-2sec-1MeV-1'])
Phi_s = np.array([integrate.trapezoid(ys[i:], xg[i:]) for i in range(len(xg))])
Phi_d = np.array([integrate.trapezoid(yd[i:], xg[i:]) for i in range(len(xg))])

norm=1e12
plt.style.use(os.path.join(REPO,'1eV','physrev.mplstyle'))
plt.figure(figsize=(8,6))
plt.plot(xg, Phi_s/norm, color='black', lw=3, label='With NC')
plt.plot(xg, Phi_d/norm, color='black', lw=3, ls='dashed', label='Without NC')
plt.scatter(edges[:-1], xo/norm, s=6, marker='o', color='0.65', zorder=4,
            label=r'Best-fit (vertex, $\Delta\chi^2=0$)')
plt.scatter(edges[:-1], xm/norm, s=10, marker='o', color='#0072B2', zorder=5,
            label=r'Best-fit (fewest steps, $\Delta\chi^2<10^{-3}$)')
plt.xscale('log'); plt.xlim(1e-1, 7.3)
plt.xlabel(r"$E_\nu$ [MeV]", fontsize=30)
plt.ylabel(nab._phi_ylabel(norm).replace('<~2~','<~7~'), fontsize=30)
plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.,10)*0.1, numticks=20))
plt.tick_params(axis='both', which='major', labelsize=23)
plt.tick_params(axis='both', which='minor', labelsize=23)
plt.legend(loc='upper right', fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'method1_bestfit_merged_{THR}.pdf'))
plt.savefig(os.path.join(RES, f'method1_bestfit_merged_{THR}.png'), dpi=115)
print('saved', f'method1_bestfit_merged_{THR}.pdf')
