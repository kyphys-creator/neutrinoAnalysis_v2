"""Method 1 trial: compress the peaks of the wide E' bins down to the level of the 2-eV bins, keeping their shape.

  T    = largest R_ij of the 2-eV-wide E' bins, computed with the uniform width Delta = (2 - E_min)/180
  P_i  = peak of row i on the uniform grid; only rows with P_i > T (the wide bins) are compressed
  g_i  = Delta * curlyR_i(E), row i on the uniform grid
  width: w(E) = Delta * min(1, min_{i: P_i > T} (T/P_i) (P_i/g_i(E))^beta)
         so a compressed row becomes r_i = T (g_i/P_i)^(1-beta): its top is exactly T, the level of
         the 2-eV peaks, and on a log axis its shape is the original one scaled by (1-beta).
         beta = 0 keeps the shape exactly (constant width under the peak), beta = 1 flattens it at T
         like equalized. Rows with P_i <= T (the 2-eV bins) are not touched, so their peaks stay at T.
  matrix: exact integral of the piecewise-linear interpolation of curlyR_table.csv (+-1 eV box resolution)

usage: python build_soft_peakT_Rij.py <1eV|5eV> [--beta=0.5] [--tag=method1softT]
output: data/edges_<tag>_<thr>.csv, data/CRmat_<tag>_<thr>_originalUnit.csv
"""
import sys, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OPT = dict(x[2:].split('=', 1) for x in sys.argv[1:] if x.startswith('--'))
ARGS = [x for x in sys.argv[1:] if not x.startswith('--')]
THR = ARGS[0] if ARGS else '1eV'
BETA = float(OPT.get('beta', 0.5))
TAG = OPT.get('tag', 'method1softT')
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
widths_eV = np.diff(BINS)
T = MU[widths_eV == 2].max()
P = MU.max(axis=1)
big = np.where(P > T*(1 + 1e-9))[0]

def width(E):
    f = 1.0
    for i in big:
        gi = delta*np.interp(E, Et, Rt[i])
        if gi > 0:
            f = min(f, (T/P[i])*(P[i]/gi)**BETA)
    return delta*f

edges = [EMIN]
while edges[-1] < Et[-1] - 1e-12:
    a = edges[-1]
    w = width(a + 0.5*width(a))                         # evaluate at the interval midpoint
    edges.append(min(a + w, Et[-1]))
edges = np.array(edges)
wk = np.diff(edges)[:-1]
k = int(np.where(wk < delta*(1 - 1e-6))[0].max()) + 1 if (wk < delta*(1 - 1e-6)).any() else 0
nt = max(1, int(round((edges[-1] - edges[k])/delta)))
edges = np.r_[edges[:k], np.linspace(edges[k], edges[-1], nt + 1)]   # equal widths in the untouched tail, no short leftover
M = matrix(edges)
w = np.diff(edges)
narrow = w < 0.99*delta
narrow[-1] = False
print(f'[{THR}] beta = {BETA}: T = {T:.3e}, N_int = {len(w)} (uniform grid {MU.shape[1]}), '
      f'narrowest {w.min()*1e3:.2f} keV = Delta/{delta/w.min():.2f} at {edges[np.argmin(w)]:.3f} MeV, '
      f'narrowed in [{edges[:-1][narrow].min():.3f}, {edges[1:][narrow].max():.3f}] MeV')
print(f'  max element / T = {M.max()/T:.3f}   (uniform grid {MU.max()/T:.2f})')
for i in big:
    j = int(np.argmax(M[i]))
    print(f'  row {BINS[i]:.0f}-{BINS[i+1]:.0f} eV: peak {P[i]/T:.2f} T on the uniform grid -> {M[i, j]/T:.3f} T at {edges[j]:.3f} MeV')

os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
np.savetxt(os.path.join(HERE, 'data', f'edges_{TAG}_{THR}.csv'), edges, delimiter=',')
np.savetxt(os.path.join(HERE, 'data', f'CRmat_{TAG}_{THR}_originalUnit.csv'), M, delimiter=',')
print(f'  saved data/edges_{TAG}_{THR}.csv, data/CRmat_{TAG}_{THR}_originalUnit.csv')
