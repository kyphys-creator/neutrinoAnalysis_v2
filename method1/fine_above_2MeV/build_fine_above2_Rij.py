"""Method 1 trial: Method 2 widths up to 2 MeV, a constant finer width above 2 MeV.

  below 2 MeV : uniform intervals of width Delta = (2 - E_min)/180 (the same 180 intervals as Method 2)
  above 2 MeV : 2-7 MeV split into N_hi equal intervals (width w = 5 MeV / N_hi < Delta).
              N_hi is the smallest number for which the peak above 2 MeV of R_ij of the last bin (81-120 eV)
              does not exceed the peak of R_ij of the first bin (the lowest E' bin).
  matrix      : exact integral of the piecewise-linear interpolation of curlyR_table.csv (+-1 eV box resolution)

usage: python build_fine_above2_Rij.py <1eV|5eV>
output: data/edges_method1f_<thr>.csv, data/CRmat_method1f_<thr>_originalUnit.csv
"""
import sys, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = sys.argv[1] if len(sys.argv) > 1 else '1eV'
EMIN = {'1eV': 0.18, '5eV': 0.41}[THR]
NB = 180
BINS = np.array([1., 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
                 43, 45, 47, 49, 51, 56, 61, 66, 71, 81, 120])[{'1eV': 0, '5eV': 2}[THR]:]

tab = pd.read_csv(os.path.join(REPO, 'Mathematica', THR, 'output', 'curlyR_table.csv'))
Et = tab.iloc[:, 0].values; Rt = tab.iloc[:, 1:].values.T
d = Rt.shape[0]
assert d == len(BINS)-1

# exact cumulative integral of the piecewise-linear interpolation, F_i(x) = int_{Et[0]}^x curlyR_i
seg = 0.5*(Rt[:, 1:] + Rt[:, :-1])*np.diff(Et)
Fk = np.concatenate([np.zeros((d, 1)), np.cumsum(seg, axis=1)], axis=1)
def F(x):
    x = np.atleast_1d(x)
    k = np.clip(np.searchsorted(Et, x, side='right') - 1, 0, len(Et) - 2)
    t = x - Et[k]; slope = (Rt[:, k+1] - Rt[:, k])/(Et[k+1] - Et[k])
    return Fk[:, k] + Rt[:, k]*t + 0.5*slope*t**2
def matrix(edges):
    return np.diff(F(edges), axis=1)

delta = (2.0 - EMIN)/NB
e_lo = np.linspace(EMIN, 2.0, NB + 1)
P_first = matrix(e_lo)[0].max()

def last_peak(N):                                   # peak of the last row above 2 MeV
    e = np.linspace(2.0, 7.0, N + 1)
    return matrix(e)[-1].max()

N = int(np.ceil(5.0/delta))
while last_peak(N) > P_first:
    N = int(np.ceil(N*last_peak(N)/P_first))
while N > 1 and last_peak(N - 1) <= P_first:
    N -= 1
edges = np.r_[e_lo, np.linspace(2.0, 7.0, N + 1)[1:]]
M = matrix(edges)
w = 5.0/N

below = e_lo[:-1]
j_lo = int(np.argmax(M[-1, :NB]))
print(f'[{THR}] Delta = {delta*1e3:.3f} keV (180 intervals below 2 MeV);  above 2 MeV: N_hi = {N}, '
      f'w = {w*1e3:.3f} keV = Delta/{delta/w:.2f};  N_int = {M.shape[1]}')
print(f'  first peak  (bin {BINS[0]:.0f}-{BINS[1]:.0f} eV) = {P_first:.3e} cm^2/kg')
print(f'  last peak   (bin {BINS[-2]:.0f}-{BINS[-1]:.0f} eV) above 2 MeV = {last_peak(N):.3e}  '
      f'(= {last_peak(N)/P_first:.3f} x first;  with width Delta it would be {last_peak(int(round(5/delta)))/P_first:.2f} x)')
print(f'  last bin just below 2 MeV (width Delta): max {M[-1, :NB].max():.3e} = {M[-1, :NB].max()/P_first:.2f} x first, '
      f'at {below[j_lo]:.3f} MeV')
print(f'  whole matrix: max element = {M.max()/P_first:.2f} x first peak')

np.savetxt(os.path.join(HERE, 'data', f'edges_method1f_{THR}.csv'), edges, delimiter=',')
np.savetxt(os.path.join(HERE, 'data', f'CRmat_method1f_{THR}_originalUnit.csv'), M, delimiter=',')
print(f'  saved data/edges_method1f_{THR}.csv, data/CRmat_method1f_{THR}_originalUnit.csv')
