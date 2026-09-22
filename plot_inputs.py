"""
Figures of the analysis *inputs* (1 eV threshold), in the plot_band_comparison /
plot_curlyR style (physrev style sheet, log E_nu axis, large labels, Okabe–Ito
colours).  Run from the repository root:

    python plot_inputs.py            # all figures
    python plot_inputs.py dnde       # one of: dnde, fig1, crmat, crmat_curlyR

Outputs (repository root):
    dNdE_histogram_interpolation.pdf   tabulated dN/dE (with / without NC) as a
                                       histogram + the piecewise-linear
                                       interpolation used in the rate calculation
    fig1_solid_histogram.pdf           fig1-solid.csv alone as a histogram
    fig1_solid_dashed_histogram.pdf    fig1-solid + fig1-dashed histograms
    CRmat180_scatter.pdf               existing 1eV/CRmat/originalUnit/CRmat180
    CRmat180_from_curlyR_scatter.pdf   Mathematica/1eV/output/CRmat180_from_curlyR

Data:
    1eV/Danny’s files/fig1-solid.csv, fig1-dashed.csv   (E_nu [MeV], dN/dE [1/fission/MeV])
    1eV/CRmat/originalUnit/CRmat180_originalUnit.csv    (31 x 180, cm^2/kg)
    Mathematica/1eV/output/CRmat180_from_curlyR_originalUnit.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

_style = os.path.join('1eV', 'physrev.mplstyle')
if os.path.exists(_style):
    plt.style.use(_style)

# Okabe–Ito (colourblind-safe)
BLUE, VERMILLION, GREEN = '#0072B2', '#D55E00', '#009E73'

# measured-energy bin edges [eV] of the 1 eV analysis (31 bins)
ER_BINS = np.array([1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23.,
                    25., 27., 29., 31., 33., 35., 37., 39., 41., 43., 45.,
                    47., 49., 51., 56., 61., 66., 71., 81., 120.])

# scatter style per bin-width class, matching Mathematica/plot_curlyR.ipynb
WIDTH_STYLE = {
    2:  dict(color='black',     marker='o', s=4,  alpha=0.4),
    5:  dict(color=BLUE,        marker='o', s=16),
    10: dict(color=VERMILLION,  marker='s', s=16),
    39: dict(color=GREEN,       marker='^', s=20),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _danny_dir():
    d = glob.glob(os.path.join('1eV', 'Danny*files'))
    if not d:
        raise FileNotFoundError("1eV/Danny’s files/ not found")
    return d[0]


def _load_fig1():
    d = _danny_dir()
    s = pd.read_csv(os.path.join(d, 'fig1-solid.csv'))
    h = pd.read_csv(os.path.join(d, 'fig1-dashed.csv'))
    return (s['MeV'].values, s['fissionMeV'].astype(float).values,
            h['MeV'].values, h['fissionMeV'].astype(float).values)


def _log_x_axis(xmin, xmax):
    plt.xscale('log')
    plt.xlim(xmin, xmax)
    plt.xlabel(r"$E_\nu$ [MeV]", fontsize=30)
    plt.gca().xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(1.0, 10) * 0.1, numticks=20))
    plt.tick_params(axis='both', which='major', labelsize=23)
    plt.tick_params(axis='both', which='minor', labelsize=23)


def _save(name):
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    print('saved', name)


# --------------------------------------------------------------------------
# 1) dN/dE: histogram of the tabulated data + interpolating curve
# --------------------------------------------------------------------------
def plot_dnde_histogram_interpolation(out='dNdE_histogram_interpolation.pdf'):
    E_s, y_s, E_d, y_d = _load_fig1()
    Ef = np.logspace(np.log10(0.1), np.log10(10), 2000)
    # np.interp == the interpolation used in NeutrinoAnalysis._calculate_integrated_flux
    f_s = np.interp(Ef, E_s, y_s, right=0)
    f_d = np.interp(Ef, E_d, y_d, right=0)

    plt.figure(figsize=(8, 6))
    # bars centred on the data points so the curve passes through the bar-top midpoints
    plt.bar(E_s, y_s, width=0.1, align='center', color=BLUE,
            edgecolor='black', linewidth=0.3, alpha=0.35, label='With NC')
    plt.bar(E_d, y_d, width=0.1, align='center', color=VERMILLION,
            edgecolor='black', linewidth=0.3, alpha=0.35, label='Without NC')
    plt.plot(Ef, f_s, color='black', lw=2.2, label='Interpolation (with NC)')
    plt.plot(Ef, f_d, color='black', lw=2.2, ls='dashed', label='Interpolation (without NC)')

    _log_x_axis(0.1, 10)
    plt.ylim(0, None)
    plt.ylabel(r"$dN_\nu/dE_\nu$ [fission$^{-1}$ MeV$^{-1}$]", fontsize=30)
    plt.legend(loc='upper right', frameon=False,
               prop={'family': 'DejaVu Sans', 'size': 16.0})
    _save(out)


# --------------------------------------------------------------------------
# 2) fig1 histograms (solid only / solid + dashed)
# --------------------------------------------------------------------------
def plot_fig1_histograms():
    E_s, y_s, E_d, y_d = _load_fig1()
    w_s = np.diff(E_s, append=E_s[-1] + 0.1)
    w_d = np.diff(E_d, append=E_d[-1] + 0.1)

    plt.figure(figsize=(8, 6))
    plt.bar(E_s, y_s, width=w_s, align='edge', color=BLUE,
            edgecolor='black', linewidth=0.3, alpha=0.8)
    _log_x_axis(0.1, 10)
    plt.ylim(0, None)
    plt.ylabel(r"$\phi_\nu$ [/fission/MeV]", fontsize=26)
    _save('fig1_solid_histogram.pdf')

    plt.figure(figsize=(8, 6))
    plt.bar(E_s, y_s, width=w_s, align='edge', color=BLUE,
            edgecolor='black', linewidth=0.3, alpha=0.7, label='With NC (solid)')
    plt.bar(E_d, y_d, width=w_d, align='edge', color=VERMILLION,
            edgecolor='black', linewidth=0.3, alpha=0.5, label='Without NC (dashed)')
    _log_x_axis(0.1, 10)
    plt.ylim(0, None)
    plt.ylabel(r"$\phi_\nu$ [/fission/MeV]", fontsize=26)
    plt.legend(loc='upper right', frameon=False, fontsize=15)
    _save('fig1_solid_dashed_histogram.pdf')


# --------------------------------------------------------------------------
# 3) response-matrix scatter plots (R_ij vs E_nu of interval j)
# --------------------------------------------------------------------------
def plot_crmat_scatter(csv, out, ylim=(1e-22, 3e-17), legend_loc='upper left'):
    M = np.loadtxt(csv, delimiter=',')
    nobs, nflux = M.shape
    if nobs != len(ER_BINS) - 1:
        raise ValueError(f'{csv}: {nobs} rows but {len(ER_BINS) - 1} bins defined')
    Ev = np.linspace(0.18, 2, nflux)          # same grid as NeutrinoAnalysis (eb)
    widths = np.diff(ER_BINS).astype(int)

    plt.figure(figsize=(8, 6))
    seen = set()
    for i in range(nobs):
        w = widths[i]
        lbl = f'Bin width {w} eV' if w not in seen else None
        seen.add(w)
        plt.scatter(Ev, M[i, :], label=lbl, **WIDTH_STYLE[w])

    _log_x_axis(0.1, 3)
    plt.yscale('log')
    plt.ylim(*ylim)
    plt.ylabel(rf"$\mathcal{{R}}_{{ij}}$ [cm$^2$/kg] ($N_{{\rm int}} = {nflux}$)", fontsize=30)
    plt.legend(loc=legend_loc, frameon=False,
               prop={'family': 'DejaVu Sans', 'size': 16.0},
               title=r"$E'$ bin", title_fontsize=22)
    _save(out)


def plot_crmat_existing():
    plot_crmat_scatter('1eV/CRmat/originalUnit/CRmat180_originalUnit.csv',
                       'CRmat180_scatter.pdf')


def plot_crmat_from_curlyR():
    plot_crmat_scatter('Mathematica/1eV/output/CRmat180_from_curlyR_originalUnit.csv',
                       'CRmat180_from_curlyR_scatter.pdf')


# --------------------------------------------------------------------------
TASKS = {
    'dnde':         plot_dnde_histogram_interpolation,
    'fig1':         plot_fig1_histograms,
    'crmat':        plot_crmat_existing,
    'crmat_curlyR': plot_crmat_from_curlyR,
}

if __name__ == '__main__':
    names = sys.argv[1:] or list(TASKS)
    for n in names:
        if n not in TASKS:
            sys.exit(f'unknown task {n!r}; choose from {list(TASKS)}')
        TASKS[n]()
