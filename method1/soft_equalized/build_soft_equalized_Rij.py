"""Method 1 trial: like equalized, but the peaks of R_ij are compressed instead of flattened.

  T     = largest R_ij of the 2-eV-wide E' bins, computed with the uniform width Delta = (2 - E_min)/180
  g(E)  = Delta * max_i curlyR_i(E), the column maximum of the uniform grid
  width = w(E) = Delta * min(1, (T/g)^alpha)
          so the column maximum becomes T^alpha * g^(1-alpha) where g > T: the peaks keep their
          position and shape and are pulled towards T. alpha = 1 is equalized (flat at T),
          alpha = 0 is the uniform grid. Where g <= T the width stays Delta.
  matrix: exact integral of the piecewise-linear interpolation of curlyR_table.csv (+-1 eV box resolution)

usage: python build_soft_equalized_Rij.py <1eV|5eV> [--alpha=0.5] [--tag=method1soft]
output: data/edges_<tag>_<thr>.csv, data/CRmat_<tag>_<thr>_originalUnit.csv
"""
import sys, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OPT = dict(x[2:].split('=', 1) for x in sys.argv[1:] if x.startswith('--'))
ARGS = [x for x in sys.argv[1:] if not x.startswith('--')]
THR = ARGS[0] if ARGS else '1eV'
ALPHA = float(OPT.get('alpha', 0.5))
TAG = OPT.get('tag', 'method1soft')
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
eU = EMIN + delta*np.arange(int(np.floor((Et[-1] - EMIN)/delta + 1e-9)) + 1)
MU = matrix(eU)
T = MU[np.diff(BINS) == 2].max()
peak = Rt.max(axis=0)                                   # max_i curlyR_i on the table points

def width(E):
    g = delta*np.interp(E, Et, peak)
    return delta*min(1.0, (T/g)**ALPHA) if g > 0 else delta

edges = [EMIN]
while edges[-1] < Et[-1] - 1e-12:
    a = edges[-1]
    w = width(a + 0.5*width(a))                         # evaluate at the interval midpoint
    edges.append(min(a + w, Et[-1]))
if edges[-1] - edges[-2] < 0.5*delta:
    del edges[-2]                                       # merge a short leftover interval at 7 MeV
edges = np.array(edges)
M = matrix(edges)
w = np.diff(edges)
cm = M.max(axis=0)
jl = int(np.argmax(M[-1]))
print(f'[{THR}] alpha = {ALPHA}: T = {T:.3e}, N_int = {len(w)} (uniform grid {MU.shape[1]}), '
      f'narrowest {w.min()*1e3:.2f} keV = Delta/{delta/w.min():.2f} at {edges[np.argmin(w)]:.3f} MeV')
print(f'  max element / T = {cm.max()/T:.2f}  (uniform grid {MU.max()/T:.2f});  '
      f'last bin ({BINS[-2]:.0f}-{BINS[-1]:.0f} eV) peaks at {edges[jl]:.3f} MeV with {M[-1, jl]/T:.2f} T')

os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
np.savetxt(os.path.join(HERE, 'data', f'edges_{TAG}_{THR}.csv'), edges, delimiter=',')
np.savetxt(os.path.join(HERE, 'data', f'CRmat_{TAG}_{THR}_originalUnit.csv'), M, delimiter=',')
print(f'  saved data/edges_{TAG}_{THR}.csv, data/CRmat_{TAG}_{THR}_originalUnit.csv')
