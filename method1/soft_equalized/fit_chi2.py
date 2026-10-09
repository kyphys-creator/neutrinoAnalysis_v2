"""Method 1 best fit: chi-square minimization, then vertex selection and Delta chi2 = 0 merge.

1. chi-square fit: minimize the Neyman chi^2 = T sum_i (data_i - (M x)_i)^2 / data_i over monotone,
   non-negative x. In the increments u_j = x_j - x_{j+1} >= 0 (u_n = x_n) this is a non-negative
   least-squares problem, solved exactly with scipy.optimize.nnls. chi2_min > 0 for noisy data;
   chi2_min = 0 for exact (noiseless) mock data.
2. vertex selection: the fitted rates mu = M x_hat are unique even though x_hat is not. Among all
   monotone, non-negative x with M x = mu, a simplex LP (scipy.optimize.linprog, HiGHS) picks the
   tail-weighted vertex, as in the main analysis. The tail weights drive the flux above ~2 MeV, which
   the data do not resolve, to zero; the NNLS solution alone instead leaves a constant floor up to 7 MeV.
3. Delta chi2 = 0 merge: adjacent constant segments are merged greedily as long as M x = mu still has
   a solution, which keeps chi2 at its minimum -> at most d-1 downward steps.

usage: python fit_chi2.py <1eV|5eV> [tag] [--data=res|ratebin7|<file.csv>]
  tag: matrix name (data/CRmat_<tag>_<thr>_originalUnit.csv). Default method1soft,
       nores = method1u_nores. Intervals from data/edges_<tag without _nores>_<thr>.csv
  --data: res = mock data with resolution, data/Ratebin7res_<thr>_originalUnit.csv
          (../mockdata/make_mockdata_res.py; built with the same kernel as curlyR_i),
          ratebin7 = original data without resolution (<thr>/Ratebin/Ratebin7_originalUnit.csv),
          or the path of any csv with one rate per E' bin (same units). Default res.
output: results/<tag>_bestfit_<thr>.npz (edges, x, mu, chi2); for a non-default --data the data name
        is appended to the tag, e.g. <tag>_ratebin7_bestfit_<thr>.npz
"""
import sys, os
import numpy as np
from scipy.optimize import linprog, nnls
from scipy.sparse import csr_matrix, lil_matrix, vstack, eye

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DATA = 'res'
DATA = next((x.split('=', 1)[1] for x in sys.argv[1:] if x.startswith('--data=')), DEFAULT_DATA)
ARGS = [x for x in sys.argv[1:] if not x.startswith('--')]
THR = ARGS[0] if ARGS else '1eV'
TAG = ARGS[1] if len(ARGS) > 1 else 'method1soft'
TAG = 'method1u_nores' if TAG == 'nores' else TAG
M1 = np.loadtxt(os.path.join(HERE, 'data', f'CRmat_{TAG}_{THR}_originalUnit.csv'), delimiter=',')
edges = np.loadtxt(os.path.join(HERE, 'data', f'edges_{TAG.replace("_nores", "")}_{THR}.csv'), delimiter=',')
n = M1.shape[1]

sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='scipy', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
conv = a.cm**2*a.sec
XS = 1e12/conv                                       # x = XS * y with y ~ O(1)
if DATA == 'res':
    data = np.loadtxt(os.path.join(HERE, 'data', f'Ratebin7res_{THR}_originalUnit.csv'))
elif DATA == 'ratebin7':
    data = np.asarray(a.Ratebin7, dtype=float)
else:
    data = np.loadtxt(os.path.join(HERE, DATA) if not os.path.isabs(DATA) else DATA)
Ms = M1*conv_mat
d = len(data)

# ---------- 1. chi-square minimization (NNLS in the increments u) ----------
sw = np.sqrt(a.T/data)                               # Neyman weights: chi2 = sum (sw*(data - mu))^2
A_u = (sw[:, None]*Ms*XS).cumsum(axis=1)             # x_j = sum_{k>=j} u_k  ->  column k = sum_{j<=k} of M
u, _ = nnls(A_u, sw*data, maxiter=50*n)
x_hat = np.cumsum(u[::-1])[::-1]*XS
mu = Ms @ x_hat
chi2 = float(np.sum((data - mu)**2*a.T/data))
print(f'[{THR}] data = {DATA}, N_int = {n}:  chi2_min = {chi2:.3e}  (T = {a.T});  '
      f'NNLS solution has {int((u > 0).sum())} nonzero increments')

# ---------- 2. vertex selection on the fitted rates mu ----------
A = csr_matrix(Ms*XS/mu[:, None])                    # rows normalized by mu
Dm = (eye(n, n, k=1) - eye(n, n)).tocsr()[:-1]
tw = 1000.0**(np.arange(n)/(n-1))

def lp(c, blocks=None):
    rows = [A]
    if blocks:
        nex = sum(e-s0 for s0, e in blocks if e > s0)
        if nex:
            ex = lil_matrix((nex, n)); r = 0
            for s0, e in blocks:
                for j in range(s0, e):
                    ex[r, j] = 1.0; ex[r, j+1] = -1.0; r += 1
            rows.append(ex.tocsr())
    Aeq = vstack(rows).tocsc()
    b = np.r_[np.ones(A.shape[0]), np.zeros(Aeq.shape[0]-A.shape[0])]
    return linprog(c, A_ub=Dm, b_ub=np.zeros(n-1), A_eq=Aeq, b_eq=b,
                   bounds=(0, None), method='highs', options={'time_limit': 60.0})

r0 = lp(tw)
if r0.status != 0:
    sys.exit(f'vertex LP failed although x_hat reproduces mu: {r0.message}')
y = r0.x

def treads(v, tol):
    b = [[0, 0]]
    for j in range(1, n):
        if abs(v[j]-v[j-1]) < tol: b[-1][1] = j
        else: b.append([j, j])
    return [tuple(t) for t in b]

# ---------- 3. Delta chi2 = 0 merge ----------
# group equal neighbours of the vertex into treads. Noisy data give genuine steps down to ~1e-10 y[0]
# (round-off is ~1e-12), so a finer tolerance is used only if the coarser grouping contradicts M x = mu
for tol_rel in (1e-6, 1e-9, 1e-11):
    blocks = treads(y, tol_rel*y[0])
    if lp(np.zeros(n), blocks).status == 0:
        break
else:
    sys.exit('no grouping of the vertex is compatible with M x = mu')
nv = len(blocks)-1; improved = True; nlp = 0
while improved:
    improved = False
    order = sorted(range(len(blocks)-1),
                   key=lambda k: min(blocks[k][1]-blocks[k][0], blocks[k+1][1]-blocks[k+1][0]))
    for k in order:
        trial = blocks[:k] + [(blocks[k][0], blocks[k+1][1])] + blocks[k+2:]
        nlp += 1
        if lp(np.zeros(n), trial).status == 0:
            blocks = trial; improved = True; break
rf = lp(tw, blocks)
if rf.status != 0:
    sys.exit(f'final LP on the merged partition failed: {rf.message}')
x = rf.x*XS
resid = np.max(np.abs(Ms@x/mu - 1))
print(f'  vertex steps {nv} -> merged steps {len(blocks)-1} (d-1 = {d-1});  '
      f'max|Mx/mu-1| = {resid:.1e};  LP tests {nlp}')
xp = x*conv
for E in [0.3, 0.5, 1.0, 1.5, 1.9, 2.5]:
    if E < edges[0]: continue
    print(f'  Phi({E:.1f} MeV) = {xp[np.searchsorted(edges, E, side="right")-1]/1e12:.3f} x 1e12 cm^-2 s^-1')
SUF = '' if DATA == DEFAULT_DATA else '_' + os.path.splitext(os.path.basename(DATA))[0]
np.savez(os.path.join(HERE, 'results', f'{TAG}{SUF}_bestfit_{THR}.npz'), edges=edges, x=xp, mu=mu, chi2=chi2)
