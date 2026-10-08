"""Numerical experiments isolating why Method 1 became stable.

Setups (all 1eV, T=3):
  S_A  adaptive grid (n=272),  data = Ratebin7,  Bkg = 0        (current Method 1)
  S_B  uniform grid  (n=272; 2-7 MeV split into 92), likewise   (control for the grid hypothesis)
  S_C  adaptive grid (n=272),  data = Ratebin7,  Bkg = RateDiff (h_i subtracted: the columns above 2 MeV
       are redundant directions with true value 0 -> do they destabilize the fit?)
  S_D  the paper's 180 columns (cut at 2 MeV), data = Ratebin7, Bkg = RateDiff (Method 2 reference)

Diagnostics:
  * distribution of the column norms (max/min)
  * singular values of the weighted matrix W^{1/2} M_s (condition number in data space; the n-d zero
    singular values are the degenerate face itself and are always there)
  * chi2/c and smoothness (number of treads) of the OSQP solution (face interior), number of treads after vertex selection
  * S_C: do the components above 2 MeV stick at 0, and does the step structure below 2 MeV match S_D?

The genuine dependence on the starting point with trust-constr (osqp is deterministic and ignores x0) is checked
  python stability_experiments.py tc_uniform   # S_B with trust-constr (slow)
separately with this command.

usage: python stability_experiments.py [osqp|tc_uniform]
"""
import sys, os, time
import numpy as np, pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else 'osqp'
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = '1eV'; EMIN = 0.18; NB = 180

# ---------- matrices ----------
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

edges_A = np.genfromtxt(os.path.join(HERE, 'data', f'edges_method1_{THR}.csv'), delimiter=',')
M_A = np.genfromtxt(os.path.join(HERE, 'data', f'CRmat_method1_{THR}_originalUnit.csv'), delimiter=',')
n_hi = len(edges_A) - 1 - NB                      # number of adaptive intervals (92)

edges_B = np.r_[edgesLow, np.linspace(2.0, 7.0, n_hi+1)[1:]]
M_B = build_matrix(edges_B)/s
M_B[:, :NB] = ref

# ---------- analysis class ----------
sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab

def make(M1, n, bkg, solver='osqp'):
    a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180',
                             GeV=0.32e16, solver=solver, T=3.0)
    conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
    a.CRmat = M1*conv_mat
    a.n = n
    a.M_matrix = a.c*a.CRmat
    a.data_vector = a.c*a.Ratebin7
    a.Bkg_vector = a.c*a.RateDiff if bkg == 'hi' else np.zeros(a.m)
    a._build_ordering_constraint()
    a._dmb_default, a._inv_d_default = a._make_dmb_inv(a.data_vector)
    a._hess_default = a._build_hessian_from(a._dmb_default, a._inv_d_default)
    a._baseline_result = None
    a.set_solver(solver)
    return a

def treads_count(x, tol_rel=1e-6):
    tol = tol_rel*max(abs(x[0]), 1e-300)
    return 1 + int((np.abs(np.diff(x)) >= tol).sum())

def diagnose(tag, M1, n, bkg, edges):
    a = make(M1, n, bkg, 'osqp')
    Ms = a.M_matrix/a.c
    cn = np.linalg.norm(Ms, axis=0)
    dmb, inv = a._dmb_default, a._inv_d_default
    W12 = np.sqrt(a.T*inv)
    sv = np.linalg.svd(W12[:, None]*Ms, compute_uv=False)
    a._backend.vertex_select = False
    r_raw = a.optimize(a.data_vector)
    a2 = make(M1, n, bkg, 'osqp'); a2._backend.vertex_select = True
    r_v = a2.optimize(a2.data_vector)
    conv = a.cm**2*a.sec
    jmin = int(np.argmin(cn))
    print(f'--- {tag}  (n={n}, bkg={bkg})')
    print(f'  col norm max/min = {cn.max()/cn.min():.1e}   '
          f'(min at j={jmin}, E={edges[jmin]:.2f} MeV)')
    print(f'  weighted SVD: s1/s_d = {sv[0]/sv[m-1]:.1e}   (d={m} nonzero)')
    print(f'  osqp raw   : chi2/c = {r_raw.fun/a.c:.2e}   treads = {treads_count(r_raw.x)}')
    print(f'  osqp vertex: chi2/c = {r_v.fun/a2.c:.2e}   treads = {treads_count(r_v.x)}   '
          f'x(Emin) = {r_v.x[0]*conv:.4e}')
    if n > NB:
        tail_raw = r_raw.x[NB:].max()*conv
        tail_v = r_v.x[NB:].max()*conv
        print(f'  flux above 2 MeV: raw max = {tail_raw:.2e}   vertex max = {tail_v:.2e}  cm^-2 s^-1')
    return r_v.x*conv

if MODE == 'osqp':
    xA = diagnose('S_A adaptive, full data      ', M_A, len(edges_A)-1, 'none', edges_A)
    xB = diagnose('S_B uniform>2MeV, full data  ', M_B, len(edges_B)-1, 'none', edges_B)
    xC = diagnose('S_C adaptive, h_i subtracted ', M_A, len(edges_A)-1, 'hi', edges_A)
    xD = diagnose('S_D truncated (Method 2)     ', ref, NB, 'hi', edgesLow)
    # does S_C below 2 MeV match S_D (Method 2)?
    dev = np.max(np.abs(xC[:NB]-xD))/max(xD[0], 1e-300)
    print(f'\nS_C vs S_D below 2 MeV: max|dx|/x[0] = {dev:.2e}')

elif MODE == 'tc_uniform':
    # S_B with trust-constr: genuine dependence on the starting point (osqp ignores x0)
    ao = make(M_B, len(edges_B)-1, 'none', 'osqp')
    scale = float(ao.optimize(ao.data_vector).x[0])   # typical scale in internal units
    a = make(M_B, len(edges_B)-1, 'none', 'scipy')
    conv = a.cm**2*a.sec
    rng = np.random.default_rng(7)
    for k in range(2):
        x0 = np.ones(a.n) if k == 0 else np.sort(rng.uniform(0, 2*scale, a.n))[::-1]
        t0 = time.time()
        r = a.optimize(a.data_vector, x0=x0)
        mu = (a.M_matrix/a.c) @ r.x
        if k == 0: mu0 = mu
        print(f'[tc S_B start {k}] {time.time()-t0:.0f}s  chi2/c = {r.fun/a.c:.2e}  '
              f'treads = {treads_count(r.x)}  max|dmu|/mu0 = {np.max(np.abs(mu/mu0-1)):.1e}',
              flush=True)
