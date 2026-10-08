"""Appendix E 掲載用の最終 best-fit 図.

exact_merge.py の結果 (results/exactmerge_<thr>.npz, d-1 段の厳密マージ解) を読み,
論文の band_comparison 様式で描く. y 軸は添字なしの Phi
(モデル化された原子炉スペクトルは ~7 MeV で終わるので Phi(7 MeV) は無視できる).

usage: python make_paper_figure.py <1eV|5eV> [tag]
出力: results/method1_bestfit_<thr>.pdf/.png  (論文へは Paper_Draft/ にコピーして使う)
      第 2 引数 tag: fit_uniform_lp.py の結果 results/<tag>_bestfit_<thr>.npz を描く
      (u = method1u) -> results/<tag>_bestfit_<thr>.*
"""
import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = sys.argv[1] if len(sys.argv) > 1 else '1eV'
RES = os.path.join(HERE, 'results')
TAG = sys.argv[2] if len(sys.argv) > 2 else 'method1u_nores'
TAG = 'method1u' if TAG == 'u' else TAG
z = np.load(os.path.join(RES, f'exactmerge_{THR}.npz' if TAG == 'method1' else f'{TAG}_bestfit_{THR}.npz'))
edges, xm = z['edges'], z['x']            # xm: マージ済み (d-1 段), cm^-2 s^-1

sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180',
                         GeV=0.32e16, solver='osqp', T=3.0)
xg = np.logspace(-2, np.log10(7.0), 4000)
ys = np.interp(xg, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'])
yd = np.interp(xg, a.fig1dashed['MeV'], a.fig1dashed['cm**-2sec-1MeV-1'])
Phi_s = np.array([integrate.trapezoid(ys[i:], xg[i:]) for i in range(len(xg))])
Phi_d = np.array([integrate.trapezoid(yd[i:], xg[i:]) for i in range(len(xg))])

norm = 1e12
plt.style.use(os.path.join(REPO, '1eV', 'physrev.mplstyle'))
plt.figure(figsize=(8, 6))
plt.plot(xg, Phi_s/norm, color='black', lw=3, label='With NC')
plt.plot(xg, Phi_d/norm, color='black', lw=3, ls='dashed', label='Without NC')
plt.scatter(edges[:-1], xm/norm, s=10, marker='o', color='#0072B2',
            zorder=5, label='No Bkg Best-fit')
plt.xscale('log'); plt.xlim(1e-1, 7.3)
plt.xlabel(r"$E_\nu$ [MeV]", fontsize=30)
plt.ylabel(r"$\Phi$ [$10^{12}$ cm$^{-2}$sec$^{-1}$]", fontsize=30)
plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1., 10)*0.1, numticks=20))
plt.tick_params(axis='both', which='major', labelsize=23)
plt.tick_params(axis='both', which='minor', labelsize=23)
plt.legend(loc='upper right', fontsize=15, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'{TAG}_bestfit_{THR}.pdf'))
plt.savefig(os.path.join(RES, f'{TAG}_bestfit_{THR}.png'), dpi=115)
nsteps = int((np.abs(np.diff(xm)) > 1e-6*max(xm[0], 1)).sum())
print(f'[{THR}] merged best fit: {nsteps} downward steps;  saved {TAG}_bestfit_{THR}.pdf/.png')
