"""Mock data for Method 1 (with resolution): the resolution kernel curlyR_i applied to the theory flux.

  N_i = int_{E_min}^{7 MeV} dE_nu curlyR_i(E_nu) Phi(E_nu),   Phi(E) = int_E^{7 MeV} phi
  curlyR_i : Mathematica/<thr>/output/curlyR_table.csv (CurlRboxG, +-1 eV box resolution)
  phi      : theory flux (With NC; fig1Solid of the analysis class, same as the theory curve in the figures)
  units    : same as Ratebin7_originalUnit.csv (the class convention M x = data)

The original Ratebin7 has no resolution (Exprate in legacy 5_Calculation.nb) and is inconsistent with curlyR_i
(-3% in the 1-3 eV bin). These data use the same kernel as curlyR_i, so they match the Method 1 matrices.
Approximation: contributions from E_nu < E_min (recoils pushed above threshold by the resolution) are left out,
as in the paper, where E_nu^min is defined neglecting the resolution. The omitted amount is printed at run time.

usage: python make_mockdata_res.py <1eV|5eV>
output: data/Ratebin7res_<thr>_originalUnit.csv, copied to ../equalized/data, ../uniform/data, ../fine_above_2MeV/data, ../soft_equalized/data, ../soft_peakT/data
"""
import os, sys, shutil
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = sys.argv[1] if len(sys.argv) > 1 else '1eV'
EMIN = {'1eV': 0.18, '5eV': 0.41}[THR]
Mn = (0.93149410372*72 - 0.0725)*1e9                      # eV (01_setup.wl)
SIG = 1.0
BINS = np.array([1., 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
                 43, 45, 47, 49, 51, 56, 61, 66, 71, 81, 120])[{'1eV': 0, '5eV': 2}[THR]:]

tab = pd.read_csv(os.path.join(REPO, 'Mathematica', THR, 'output', 'curlyR_table.csv'))
Et = tab.iloc[:, 0].values; Rt = tab.iloc[:, 1:].values.T
d = Rt.shape[0]
assert d == len(BINS) - 1 and abs(Et[0] - EMIN) < 1e-9

sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='scipy', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
conv = a.cm**2*a.sec

Ef = np.linspace(0.01, 7.0, 699001)
phi = np.interp(Ef, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'], left=0, right=0)
Phi = np.r_[np.cumsum((0.5*(phi[1:]+phi[:-1])*np.diff(Ef))[::-1])[::-1], 0.0]   # int_E^7

m = Ef >= EMIN
Rf = np.array([np.interp(Ef[m], Et, Rt[i]) for i in range(d)])
N = np.trapezoid(Rf*Phi[m][None, :], Ef[m], axis=1)*conv_mat/conv

old = np.asarray(a.Ratebin7, dtype=float)
r = N/old
print(f'[{THR}] new / old Ratebin7: {BINS[0]:.0f}-{BINS[1]:.0f} eV {r[0]:.4f}, '
      f'{BINS[1]:.0f}-{BINS[2]:.0f} eV {r[1]:.4f}, ..., {BINS[-2]:.0f}-{BINS[-1]:.0f} eV {r[-1]:.4f}')

# omitted contribution from E_nu < E_min (estimated by rebuilding the table's box kernel analytically)
def ERmax(E):
    E = E*1e6
    return 2*E**2/(Mn + 2*E)
ERg = np.r_[np.geomspace(1e-4, 0.999, 3000), np.linspace(1.0, 130.0, 52000)]
def box(er, e1, e2):
    g = lambda u: u*(u > 0)
    return ((g(e2-er+SIG) - g(e2-er-SIG)) - (g(e1-er+SIG) - g(e1-er-SIG)))/(2*SIG)
def kernel_box(E, i):
    w = ERg*box(ERg, BINS[i], BINS[i+1])
    cw = np.r_[0.0, np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(ERg))]
    return np.interp(ERmax(E), ERg, cw)/E**3
hiE = Et >= 3.0
lo = ~m
for i in range(2):
    Ci = np.median(Rt[i, hiE]/kernel_box(Et[hiE], i))
    miss = np.trapezoid(Ci*kernel_box(Ef[lo], i)*Phi[lo], Ef[lo])*conv_mat/conv
    print(f'  omitted E_nu < E_min contribution, bin {BINS[i]:.0f}-{BINS[i+1]:.0f} eV: {miss/N[i]:.2%}')

out = os.path.join(HERE, 'data', f'Ratebin7res_{THR}_originalUnit.csv')
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savetxt(out, N)
for dst in ('equalized', 'uniform', 'fine_above_2MeV', 'soft_equalized', 'soft_peakT'):
    shutil.copy(out, os.path.join(os.path.dirname(HERE), dst, 'data'))
print(f'  saved data/Ratebin7res_{THR}_originalUnit.csv (+ copies in equalized/data, uniform/data, fine_above_2MeV/data, soft_equalized/data, soft_peakT/data)')
