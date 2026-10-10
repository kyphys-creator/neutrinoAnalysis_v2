"""Why the old shortcut and the chi-square minimization give the same best fit (equalized grid).

The shortcut (fit_uniform_lp.py, up to commit 1ae0fd4) solved sum_j R_ij Phi_j = data_i exactly
instead of minimizing chi^2; its best fits are kept as data/shortcut_bestfit_<thr>.npz.
For the noiseless mock data chi2_min is zero within double precision, so both give the same fit.

(a)(b) best fits of the shortcut and of fit_chi2.py, 1 eV and 5 eV.
(c)    |pull_i| = |data_i - mu_i| / sqrt(data_i/T) per E' bin for the chi^2 minimization, on a log axis:
       noiseless mock data (chi2_min ~ 1e-26) vs the same data with 1% fluctuations (chi2_min > 0).
       The band is the double-precision rounding of the rates, eps*sqrt(T data_i).

usage: python compare_shortcut_vs_chi2.py
output: results/shortcut_vs_chi2.pdf/.png
"""
import os, sys, importlib
import numpy as np
from scipy.optimize import nnls
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EPS = np.finfo(float).eps

def sci(v):
    m, e = f'{v:.0e}'.split('e')
    return rf'{m}\times10^{{{int(e)}}}'

def chi2_fit(Ms, data, T, XS):
    sw = np.sqrt(T/data)
    u, _ = nnls((sw[:, None]*Ms*XS).cumsum(axis=1), sw*data, maxiter=50*Ms.shape[1])
    mu = Ms @ (np.cumsum(u[::-1])[::-1]*XS)
    return mu, float(np.sum((data - mu)**2*T/data))

plt.style.use(os.path.join(REPO, '1eV', 'physrev.mplstyle'))
fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
pulls = {}
for ax, thr in zip(axes[:2], ('1eV', '5eV')):
    sys.path.insert(0, os.path.join(REPO, thr)); os.chdir(os.path.join(REPO, thr))
    import neutrino_analysis_band as nab; importlib.reload(nab)
    a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='scipy', T=3.0)
    os.chdir(HERE); sys.path.pop(0)
    conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr; conv = a.cm**2*a.sec; XS = 1e12/conv
    M1 = np.loadtxt(os.path.join(HERE, 'data', f'CRmat_method1eq_{thr}_originalUnit.csv'), delimiter=',')
    data = np.loadtxt(os.path.join(HERE, 'data', f'Ratebin7res_{thr}_originalUnit.csv'))
    S = np.load(os.path.join(HERE, 'data', f'shortcut_bestfit_{thr}.npz'))
    C = np.load(os.path.join(HERE, 'results', f'method1eq_bestfit_{thr}.npz'))
    dev = np.max(np.abs(C['x'] - S['x']))/S['x'][0]
    print(f'[{thr}] chi2_min = {float(C["chi2"]):.1e};  max|x_chi2 - x_shortcut|/x[0] = {dev:.1e}')

    Ef = np.logspace(-2, np.log10(7.0), 4000)
    ph = np.interp(Ef, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'])
    Phi = np.array([np.trapezoid(ph[i:], Ef[i:]) for i in range(len(Ef))])
    ax.plot(Ef, Phi/1e12, color='black', lw=2, label='true flux (With NC)')
    ax.scatter(S['edges'][:-1], S['x']/1e12, s=60, facecolors='none', edgecolors='#D55E00', lw=1.2,
               label='old shortcut: solve $\\sum_j R_{ij}\\Phi_j = {\\rm data}_i$')
    ax.scatter(C['edges'][:-1], C['x']/1e12, s=8, color='#0072B2', zorder=5,
               label=f'$\\chi^2$ minimization ($\\chi^2_{{\\min}} = {sci(float(C["chi2"]))}$)')
    ax.set_xscale('log'); ax.set_xlim(0.15, 7.3); ax.set_ylim(-0.1, 2.1)
    ax.set_xlabel(r'$E_\nu$ [MeV]', fontsize=20); ax.tick_params(labelsize=14)
    ax.set_title(rf"({'a' if thr == '1eV' else 'b'}) $E'_{{\rm thr}} = {thr[0]}$ eV:  max difference / $\Phi(E_{{\min}})$ = {dev:.0e}", fontsize=14)
    ax.legend(loc='upper right', frameon=False, fontsize=11)

    rng = np.random.default_rng(3)
    noisy = data*(1 + 0.01*rng.standard_normal(len(data)))
    mu_n, chi_n = chi2_fit(M1*conv_mat, noisy, a.T, XS)
    pulls[thr] = dict(clean=np.abs(data - C['mu'])*np.sqrt(a.T/data), noisy=np.abs(noisy - mu_n)*np.sqrt(a.T/noisy),
                      floor=EPS*np.sqrt(a.T*data), chi_clean=float(C['chi2']), chi_noisy=chi_n)
    print(f'[{thr}] 1% fluctuations: chi2_min = {chi_n:.2f}')
axes[0].set_ylabel(r'$\Phi$ [$10^{12}$ cm$^{-2}$s$^{-1}$]', fontsize=20)

ax = axes[2]
for thr, mk in (('1eV', 'o'), ('5eV', 's')):
    p = pulls[thr]; i = np.arange(1, len(p['clean']) + 1)
    ax.semilogy(i, np.maximum(p['clean'], 1e-17), mk, color='#0072B2', ms=6, mfc='none' if thr == '5eV' else '#0072B2',
                label=f'noiseless mock data, {thr[0]} eV ($\\chi^2_{{\\min}} = {sci(p["chi_clean"])}$)')
    ax.semilogy(i, p['noisy'], mk, color='#D55E00', ms=6, mfc='none' if thr == '5eV' else '#D55E00',
                label=f'1% fluctuations, {thr[0]} eV ($\\chi^2_{{\\min}}$ = {p["chi_noisy"]:.2f})')
f = pulls['1eV']['floor']
ax.fill_between(np.arange(1, len(f) + 1), f/3, f*3, color='0.8', label='double-precision rounding of the rates')
ax.axhline(1.0, color='black', lw=1, ls='--')
ax.text(31.5, 1.3, r'1$\sigma$', fontsize=13, ha='right')
ax.text(0.5, 2e-17, 'exactly zero', fontsize=10, color='0.35', va='bottom')
ax.set_ylim(1e-17, 1e2); ax.set_xlim(0, 32)
ax.set_xlabel(r"$E'$ bin $i$", fontsize=20); ax.set_ylabel(r'$|{\rm data}_i - \mu_i|\,/\,\sigma_i$', fontsize=20)
ax.tick_params(labelsize=14)
ax.set_title('(c) residual per bin of the $\\chi^2$ minimization', fontsize=14)
ax.legend(loc='center right', frameon=False, fontsize=10.5)
plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(HERE, 'results', f'shortcut_vs_chi2.{ext}'), dpi=115)
print('saved results/shortcut_vs_chi2.pdf/.png')
