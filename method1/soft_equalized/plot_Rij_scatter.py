"""Scatter plot of the Method 1 response matrix R_ij (same style as CRmat180_scatter.pdf in the paper).

The matrices are built from curlyR_i. Select with tag:
  method1u  : uniform intervals with the below-2-MeV width, up to 7 MeV (build_uniform_Rij.py)
  method1eq : adaptive grid that equalizes the size of R_ij (build_equalized_Rij.py)
  method1soft: like method1eq, but peaks compressed as T^alpha g^(1-alpha) instead of flattened
              (build_soft_equalized_Rij.py)
R_ij is plotted against the lower edge E_nu^j of interval j. Colour/marker = E' bin width
(WIDTH_STYLE in figures/plot_inputs.py). Dashed line: 2 MeV; dotted line: largest element T of the 2-eV bins.

usage: python plot_Rij_scatter.py [1eV|5eV ...] [tag]   (default: both thresholds, method1soft)
output: results/<tag>_Rij_scatter_<thr>.pdf/.png  (method1_Rij_scatter_<thr> for method1u)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'figures'))
from plot_inputs import ER_BINS, WIDTH_STYLE, _log_x_axis, plt

ER_FIRST = {'1eV': 0, '5eV': 2}        # 5 eV has no first two bins (1-3, 3-5 eV)


def plot_Rij_scatter(thr, tag='method1soft', ylim=(1e-23, 3e-17)):
    M = np.loadtxt(os.path.join(HERE, 'data', f'CRmat_{tag}_{thr}_originalUnit.csv'), delimiter=',')
    edges = np.loadtxt(os.path.join(HERE, 'data', f'edges_{tag.replace("_nores", "")}_{thr}.csv'), delimiter=',')
    nobs, nflux = M.shape
    bins = ER_BINS[ER_FIRST[thr]:]
    if nobs != len(bins) - 1 or nflux != len(edges) - 1:
        raise ValueError(f'{thr}: matrix {M.shape} vs {len(bins) - 1} bins, {len(edges) - 1} intervals')
    widths = np.diff(bins).astype(int)

    plt.figure(figsize=(8, 6))
    seen = set()
    for i in range(nobs):
        w = widths[i]
        lbl = f'Bin width {w} eV' if w not in seen else None
        seen.add(w)
        plt.scatter(edges[:-1], np.where(M[i] > 0, M[i], np.nan), label=lbl, **WIDTH_STYLE[w])
    plt.axvline(2.0, color='0.5', lw=1.0, ls='--', zorder=0)
    plt.axhline(M[widths == 2].max(), color='0.3', lw=1.0, ls=':', zorder=0)

    _log_x_axis(0.1, 8)
    plt.yscale('log')
    plt.ylim(*ylim)
    plt.ylabel(rf"$\mathcal{{R}}_{{ij}}$ [cm$^2$/kg] ($N_{{\rm int}} = {nflux}$)", fontsize=30)
    plt.legend(loc='upper left', frameon=False,
               prop={'family': 'DejaVu Sans', 'size': 16.0},
               title=rf"$E'$ bin ($E'_{{\rm thr}} = {thr[0]}$ eV)", title_fontsize=20)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        name = 'method1' if tag == 'method1u' else tag
        out = os.path.join(HERE, 'results', f'{name}_Rij_scatter_{thr}.{ext}')
        plt.savefig(out, bbox_inches='tight', dpi=115)
    plt.close()
    tiny = (M > 0) & (M < ylim[0])
    noise = np.abs(M) < 1e-12*M.max()
    print(f'[{thr}] saved {name}_Rij_scatter_{thr}.pdf/.png  ({M.shape[0]} x {nflux});  '
          f'not shown: {(tiny & ~noise).sum()} elements in [{1e-12*M.max():.0e}, {ylim[0]:.0e}) '
          f'(near kinematic onset), {noise.sum()} round-off elements |R_ij| < 1e-12 max')


if __name__ == '__main__':
    thrs = [a for a in sys.argv[1:] if a in ('1eV', '5eV')] or ['1eV', '5eV']
    tag = next((a for a in sys.argv[1:] if a.startswith('method1')), 'method1soft')
    for thr in thrs:
        plot_Rij_scatter(thr, tag)
