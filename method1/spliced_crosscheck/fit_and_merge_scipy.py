"""Pure-scipy Method 1 pipeline: fit + vertex selection + exact merge.

Version without cvxpy/osqp:
  * fit                -- the analysis class with solver='scipy' (trust-constr, as in the main paper)
  * vertex (crossover) -- scipy.optimize.linprog (HiGHS simplex):
                    tail-weighted minimization keeping M x = mu exactly, to a vertex of the monotone polytope
  * exact merge        -- greedy merging of adjacent treads at Delta chi2 = 0 (M x = mu exactly),
                    each tested for feasibility with linprog -> d-1 steps
Numerical note: the equality constraints are imposed on the scaled variable y = x/XS (XS = x[0]). With the raw M_s x = mu
the coefficients are ~1e-7 against variables ~1e8, and HiGHS's absolute tolerance would break the fit
(same reason as the note in neutrino_analysis_band._OSQPBackend._build_lp).

usage: python fit_and_merge_scipy.py <1eV|5eV>
output: overwrites results/exactmerge_<thr>.npz (edges, x, x_vertex) and
      prints a comparison with the old cvxpy result
"""
import sys, os, time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, vstack, eye

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = sys.argv[1] if len(sys.argv) > 1 else '1eV'
RES = os.path.join(HERE, 'results')

M1 = np.genfromtxt(os.path.join(HERE, 'data', f'CRmat_method1_{THR}_originalUnit.csv'), delimiter=',')
edges = np.genfromtxt(os.path.join(HERE, 'data', f'edges_method1_{THR}.csv'), delimiter=',')
n = len(edges) - 1

sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab

a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180',
                         GeV=0.32e16, solver='scipy', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
a.CRmat = M1*conv_mat
a.n = n
a.M_matrix = a.c*a.CRmat
a.data_vector = a.c*a.Ratebin7          # neutrinos above 2 MeV are signal too; no h_i subtraction
a.Bkg_vector = np.zeros(a.m)
a._build_ordering_constraint()
a._dmb_default, a._inv_d_default = a._make_dmb_inv(a.data_vector)
a._hess_default = a._build_hessian_from(a._dmb_default, a._inv_d_default)
a._baseline_result = None
a.set_solver('scipy')

t0 = time.time()
res = a.optimize(a.data_vector)
conv = a.cm**2*a.sec
x_int = res.x.copy()                    # trust-constr solution in the interior of the optimal face
chi2 = lambda x: float(np.sum((a.data_vector - a.M_matrix@x)**2
                       / np.where(a.data_vector > 0, a.data_vector, 1)))/a.c
print(f'[{THR}] scipy trust-constr: {time.time()-t0:.1f}s  chi2/c = {chi2(x_int):.3e}  '
      f'x(Emin) = {x_int[0]*conv:.4e} cm^-2 s^-1')

# ---------- vertex selection + merge (all linprog / HiGHS) ----------
Ms = a.M_matrix/a.c
XS = float(x_int[0])
A = Ms*XS                               # equality-constraint matrix (scaled)
mu = Ms @ x_int
b_eq = mu
tw = 1000.0**(np.arange(n)/max(n-1, 1)) # tail weights: push the high-energy end to 0
# monotonicity y[j+1] <= y[j]: sparse (n-1) x n matrix
Dmono = (eye(n, n, k=1) - eye(n, n)).tocsr()[:-1]

def solve_lp(blocks=None, feas_only=False):
    """LP with equality constraints between adjacent intervals inside each block. c=0 if feas_only."""
    A_eq_rows = [A]
    if blocks:
        extra = lil_matrix((sum(e-s for s, e in blocks if e > s), n))
        r = 0
        for s, e in blocks:
            for j in range(s, e):
                extra[r, j] = 1.0; extra[r, j+1] = -1.0; r += 1
        if r: A_eq_rows.append(extra.tocsr())
    A_eq = vstack([lil_matrix(m) if not hasattr(m, 'tocsr') else m for m in A_eq_rows]).tocsc()
    b = np.r_[b_eq, np.zeros(A_eq.shape[0]-len(b_eq))]
    c = np.zeros(n) if feas_only else tw
    return linprog(c, A_ub=Dmono, b_ub=np.zeros(n-1), A_eq=A_eq, b_eq=b,
                   bounds=(0, None), method='highs',
                   options={'time_limit': 30.0, 'presolve': True})

r0 = solve_lp()                          # crossover: to a vertex with the same fit
assert r0.status == 0, r0.message
x_vertex = r0.x*XS
tol = 1e-6*x_vertex[0]

def treads(x):
    b = [[0, 0]]
    for j in range(1, n):
        if abs(x[j]-x[j-1]) < tol: b[-1][1] = j
        else: b.append([j, j])
    return [tuple(t) for t in b]

blocks = treads(x_vertex)
print(f'[vertex] chi2/c = {chi2(x_vertex):.3e}  treads = {len(blocks)} '
      f'(downward steps {len(blocks)-1})')

nlp = 0
improved = True
while improved:
    improved = False
    order = sorted(range(len(blocks)-1),
                   key=lambda k: min(blocks[k][1]-blocks[k][0], blocks[k+1][1]-blocks[k+1][0]))
    for k in order:
        trial = blocks[:k] + [(blocks[k][0], blocks[k+1][1])] + blocks[k+2:]
        nlp += 1
        if solve_lp(trial, feas_only=True).status == 0:
            blocks = trial; improved = True; break

rf = solve_lp(blocks)                    # tail-weighted representative within the final partition
assert rf.status == 0, rf.message
x_fit = rf.x*XS
d = a.m
print(f'[merge] d = {d}: steps -> {len(blocks)-1}  (d-1 = {d-1});  '
      f'chi2/c = {chi2(x_fit):.3e};  LP tests {nlp}')

# ---------- compare with the old cvxpy result and overwrite ----------
out = os.path.join(RES, f'exactmerge_{THR}.npz')
if os.path.exists(out):
    z = np.load(out)
    dev = np.max(np.abs(x_fit*conv - z['x']))/(z['x'][0])
    print(f'[compare] vs cvxpy pipeline: max|dx|/x[0] = {dev:.2e}')
np.savez(out, edges=edges, x=x_fit*conv, x_vertex=x_vertex*conv)
print(f'saved {out}')
