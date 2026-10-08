"""grid_dependence_test.py の結果の図 (Danny への回答用).

(a)(b) 閾値固定で 2 MeV 超の区間数 N_hi を変えた best fit と, 現行グリッドの
       Delta chi2 = 0 帯 (M x = data を厳密に満たす単調解の各 E での範囲).
(c)    1 eV と 5 eV の best fit を両方の Delta chi2 = 0 帯と重ねた拡大図 (0.5 MeV 付近).

usage: python plot_grid_dependence.py   (grid_dependence_test.py を両閾値で実行した後)
"""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, 'results')
plt.style.use(os.path.join(REPO, '1eV', 'physrev.mplstyle'))
plt.rcParams.update({'axes.labelsize': 15, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                     'legend.fontsize': 10.5})

Z = {t: np.load(os.path.join(RES, f'grid_dependence_{t}.npz')) for t in ('1eV', '5eV')}
NHI = [46, 92, 106, 184]
STYLE = {46: dict(color='#0072B2', lw=5.0, ls='-', alpha=0.45),
         92: dict(color='#E69F00', lw=3.2, ls='-'),
         106: dict(color='#009E73', lw=1.8, ls='--'),
         184: dict(color='#CC79A7', lw=1.0, ls=':')}
BAND = {'1eV': '#0072B2', '5eV': '#D55E00'}
N = 1e12

def stairs(ax, edges, x, **kw):
    ax.stairs(x/N, edges, baseline=None, **kw)

def band(ax, edges, b, color, label, alpha=0.18):
    ax.stairs(b[:, 1]/N, edges, baseline=b[:, 0]/N, fill=True, color=color,
              alpha=alpha, lw=0, label=label)

fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
for ax, t, lab in zip(axes[:2], ('1eV', '5eV'), ('(a)', '(b)')):
    z = Z[t]
    band(ax, z['edges0'], z['band'], '0.55', r'$\Delta\chi^2=0$ band', alpha=0.30)
    ax.plot(z['xg'], z['Phi_true']/N, color='black', lw=1.2, label='true flux (with NC)')
    for k in NHI:
        e, x = z[f'edges_{k}'], z[f'x_{k}']
        stairs(ax, e, x, label=rf'$N_{{\rm hi}}={k}$ ($N_{{\rm int}}={len(e)-1}$)', **STYLE[k])
    ax.axvline(2.0, color='0.4', lw=0.6, ls='--')
    ax.set_xscale('log'); ax.set_xlim(0.15, 7.2); ax.set_ylim(0, 2.4)
    ax.set_xlabel(r'$E_\nu$ [MeV]')
    ax.set_ylabel(r'$\Phi$ [$10^{12}$ cm$^{-2}$s$^{-1}$]')
    ax.set_title(rf'{lab} $E^\prime_{{\rm thr}}={t[0]}$ eV: vary $N_{{\rm hi}}$ above 2 MeV',
                 fontsize=13)
    ax.legend(loc='upper right', frameon=False)

ax = axes[2]
for t in ('1eV', '5eV'):
    z = Z[t]
    band(ax, z['edges0'], z['band'], BAND[t], rf'$\Delta\chi^2=0$ band, {t[0]} eV')
for t, ls in (('1eV', '-'), ('5eV', '--')):
    z = Z[t]
    stairs(ax, z['edges0'], z['x0'], color=BAND[t], lw=2.0, ls=ls, label=f'best fit, {t[0]} eV')
z = Z['1eV']
ax.plot(z['xg'], z['Phi_true']/N, color='black', lw=1.2, label='true flux (with NC)')
ax.axvline(0.5, color='0.3', lw=0.7, ls=':')
ax.set_xlim(0.3, 1.0); ax.set_ylim(1.05, 2.0)
ax.set_xlabel(r'$E_\nu$ [MeV]')
ax.set_ylabel(r'$\Phi$ [$10^{12}$ cm$^{-2}$s$^{-1}$]')
ax.set_title(r'(c) 1 eV vs 5 eV near 0.5 MeV', fontsize=13)
ax.legend(loc='upper right', frameon=False)

plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(RES, f'method1_grid_dependence.{ext}'), dpi=130)
print('saved results/method1_grid_dependence.pdf/.png')
