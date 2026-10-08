"""Danny's question: at a fixed threshold, does the best fit change if only the number of intervals above 2 MeV changes?

For noiseless mock data chi2_min = 0, so the solution set is exactly
    { x : x >= 0, x_j >= x_{j+1}, M x = data }
Hence it can be handled exactly with scipy.optimize.linprog alone:
  * best fit  -- same rule as fit_and_merge_scipy.py (tail-weighted vertex + Delta chi2 = 0 merge)
  * Delta chi2 = 0 band -- min / max of x_j at each E (two LPs)
The equality rows are normalized by data_i, so HiGHS's absolute tolerance 1e-7 becomes a relative error on each rate.

Grid: 180 uniform intervals below 2 MeV as in the paper (published CRmat180 columns),
above 2 MeV N_hi intervals of equal area of sum_i R_i (N_hi varied). Reference: the current grid.

usage: python grid_dependence_test.py <1eV|5eV>
"""
import sys, os
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix, vstack, eye
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = sys.argv[1] if len(sys.argv) > 1 else '1eV'
EMIN = {'1eV': 0.18, '5eV': 0.41}[THR]
NB = 180
PROBE_E = [0.3, 0.45, 0.5, 0.6, 0.8, 1.0, 1.5, 1.9]

kn = pd.read_csv(os.path.join(HERE, 'data', f'curlyR_knots_{THR}.csv'))
Ek = kn.iloc[:, 0].values
K = kn.iloc[:, 1:].values.T
m = K.shape[0]

def pl_integral(y, a, b):
    xs = np.r_[a, Ek[(Ek > a) & (Ek < b)], b]
    ys = np.interp(xs, Ek, y)
    return np.sum(0.5*(ys[1:]+ys[:-1])*np.diff(xs))

def build_matrix(edges):
    M = np.zeros((m, len(edges)-1))
    for j in range(len(edges)-1):
        for i in range(m):
            M[i, j] = pl_integral(K[i], edges[j], edges[j+1])
    return M

edgesLow = np.linspace(EMIN, 2.0, NB+1)
ref = np.genfromtxt(os.path.join(REPO, THR, 'CRmat', 'originalUnit', f'CRmat{NB}_originalUnit.csv'),
                    delimiter=',')
s = np.median(build_matrix(edgesLow).sum(1)/ref.sum(1))
S = K.sum(axis=0)

def equal_area_edges(N):
    total = pl_integral(S, 2.0, 7.0)
    out = [2.0]
    for k in range(1, N):
        target = k*total/N
        lo, hi = out[-1], 7.0
        for _ in range(60):
            mid = 0.5*(lo+hi)
            (lo, hi) = (mid, hi) if pl_integral(S, 2.0, mid) < target else (lo, mid)
        out.append(0.5*(lo+hi))
    out.append(7.0)
    return np.array(out)

def grid(N_hi):
    edges = np.r_[edgesLow, equal_area_edges(N_hi)[1:]]
    M = build_matrix(edges)/s
    M[:, :NB] = ref
    return edges, M

sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180',
                         GeV=0.32e16, solver='scipy', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
conv = a.cm**2*a.sec
XS = 1e12/conv                                  # y = x/XS ~ O(1)

def solution_set(M1, data_s):
    """LP pieces for {x>=0, monotone, M x = data} (rows normalized by data)."""
    Ms = (M1*conv_mat)
    A = csr_matrix((Ms*XS)/data_s[:, None])
    n = M1.shape[1]
    Dm = (eye(n, n, k=1) - eye(n, n)).tocsr()[:-1]
    return A, Dm, n

def lp(A, Dm, n, c, blocks=None):
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

def best_fit(M1, data_s):
    A, Dm, n = solution_set(M1, data_s)
    tw = 1000.0**(np.arange(n)/max(n-1, 1))
    r0 = lp(A, Dm, n, tw)
    if r0.status != 0:
        raise RuntimeError(f'M x = data infeasible (chi2_min > 0): {r0.message}')
    y = r0.x; tol = 1e-6*y[0]
    def treads(v):
        b = [[0, 0]]
        for j in range(1, n):
            if abs(v[j]-v[j-1]) < tol: b[-1][1] = j
            else: b.append([j, j])
        return [tuple(t) for t in b]
    blocks = treads(y); improved = True
    while improved:
        improved = False
        order = sorted(range(len(blocks)-1),
                       key=lambda k: min(blocks[k][1]-blocks[k][0], blocks[k+1][1]-blocks[k+1][0]))
        for k in order:
            trial = blocks[:k] + [(blocks[k][0], blocks[k+1][1])] + blocks[k+2:]
            if lp(A, Dm, n, np.zeros(n), trial).status == 0:
                blocks = trial; improved = True; break
    rf = lp(A, Dm, n, tw, blocks)
    x = rf.x*XS
    resid = np.max(np.abs((M1*conv_mat)@x/data_s - 1))
    return x*conv, len(blocks)-1, resid, (A, Dm, n)

def band_at(parts, j):
    A, Dm, n = parts
    c = np.zeros(n); c[j] = 1.0
    lo = lp(A, Dm, n, c).fun*XS*conv
    hi = -lp(A, Dm, n, -c).fun*XS*conv
    return lo, hi

def value_at(edges, x, E):
    return x[np.searchsorted(edges, E, side='right')-1]

def step_E(edges, x):
    jj = np.r_[0, 1+np.where(np.abs(np.diff(x)) >= 1e-6*x[0])[0]]
    return edges[jj]

data7 = np.asarray(a.Ratebin7, dtype=float)     # Method 1: the whole rate is signal (the class's c cancels on both sides)

# ---------- 1. reference: current grid (check that the saved scipy-pipeline result is reproduced) ----------
edges0 = np.genfromtxt(os.path.join(HERE, 'data', f'edges_method1_{THR}.csv'), delimiter=',')
M0 = np.genfromtxt(os.path.join(HERE, 'data', f'CRmat_method1_{THR}_originalUnit.csv'), delimiter=',')
x0, ns0, res0, parts0 = best_fit(M0, data7)
z = np.load(os.path.join(HERE, 'data', f'exactmerge_{THR}.npz'))
print(f'[{THR}] d = {m}, nominal grid N_int = {len(edges0)-1} (N_hi = {len(edges0)-1-NB}): '
      f'steps {ns0}, max|Mx/data-1| = {res0:.1e}')
print(f'  vs saved trust-constr pipeline: max|dx|/x[0] = {np.max(np.abs(x0-z["x"]))/z["x"][0]:.1e}')

# ---------- 2. vary N_hi ----------
below = edges0[:-1] < 2.0
print(f'\n  {"N_hi":>5} {"N_int":>5} {"steps":>5}  {"max|dPhi|/Phi(Emin), E<2MeV":>28}  '
      f'{"same step E (<2MeV)":>20}')
sE0 = step_E(edges0, x0); sE0 = sE0[sE0 < 2.0]
results = {}
for N_hi in [46, 92, 106, 184]:
    e, M = grid(N_hi)
    x, ns, res, _ = best_fit(M, data7)
    dev = np.max(np.abs(x[:NB] - x0[:NB]))/x0[0]
    sE = step_E(e, x); sE = sE[sE < 2.0]
    same = len(sE) == len(sE0) and np.allclose(sE, sE0)
    results[N_hi] = (e, x)
    print(f'  {N_hi:>5} {len(e)-1:>5} {ns:>5}  {dev:>28.1e}  {str(same):>20}')

# ---------- 3. probe points: best fit, Delta chi2 = 0 band, true value ----------
xg = np.logspace(-2, np.log10(7.0), 4000)
ys = np.interp(xg, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'])
Phi_true = np.array([integrate.trapezoid(ys[i:], xg[i:]) for i in range(len(xg))])
print(f'\n  E [MeV]  best fit   Delta chi2=0 band     truth   (units 1e12 cm^-2 s^-1)')
probe = {}
for E in PROBE_E:
    if E < EMIN: continue
    j = np.searchsorted(edges0, E, side='right')-1
    lo, hi = band_at(parts0, j)
    v = value_at(edges0, x0, E)
    tr = np.interp(E, xg, Phi_true)
    probe[E] = (v, lo, hi, tr)
    print(f'  {E:6.2f}   {v/1e12:7.3f}   [{lo/1e12:6.3f}, {hi/1e12:6.3f}]   {tr/1e12:6.3f}')

np.savez(os.path.join(HERE, 'results', f'grid_dependence_{THR}.npz'),
         edges0=edges0, x0=x0,
         probe_E=np.array(list(probe.keys())), probe=np.array(list(probe.values())),
         **{f'edges_{k}': v[0] for k, v in results.items()},
         **{f'x_{k}': v[1] for k, v in results.items()})

# ---------- 4. Delta chi2 = 0 band on all intervals (current grid) and Method 2 (same pipeline as the paper figures) ----------
band = np.array([band_at(parts0, j) for j in range(len(edges0)-1)])
a2 = nab.NeutrinoAnalysis(background_scenario='none', intervals='180',
                          GeV=0.32e16, solver='osqp', T=3.0)
a2._backend.vertex_select = True
x2 = a2.optimize(a2.data_vector).x*conv + np.interp(2.0, xg, Phi_true)   # Phi = deltaPhi + Phi(2 MeV)
print(f'\n  Method 2 (paper pipeline) at 0.5 MeV: {value_at(edgesLow, x2, 0.5)/1e12:.3f}   '
      f'Method 1: {value_at(edges0, x0, 0.5)/1e12:.3f}')
for N_hi, (e, x) in results.items():
    print(f'  Method 1 N_hi={N_hi:>3} at 0.5 MeV: {value_at(e, x, 0.5)/1e12:.3f}')
np.savez(os.path.join(HERE, 'results', f'grid_dependence_{THR}.npz'),
         edges0=edges0, x0=x0, band=band, xg=xg, Phi_true=Phi_true,
         edges_m2=edgesLow, x_m2=x2,
         probe_E=np.array(list(probe.keys())), probe=np.array(list(probe.values())),
         **{f'edges_{k}': v[0] for k, v in results.items()},
         **{f'x_{k}': v[1] for k, v in results.items()})
print('saved results/grid_dependence_%s.npz' % THR)
