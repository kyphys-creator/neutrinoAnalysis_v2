"""一様区間・curlyR 由来の行列 (build_uniform_Rij.py) で Method 1 の best fit を確認.

grid_dependence_test.py と同じ: 無ノイズデータでは chi2_min = 0 なので
{x >= 0, 単調, M x = data} を scipy.optimize.linprog で直接扱い,
tail 重み頂点 + Delta chi2 = 0 マージ. 既存結果は上書きしない.

usage: python fit_uniform_lp.py <1eV|5eV> [tag]
  tag: 行列の名前 (data/CRmat_<tag>_<thr>_originalUnit.csv). 省略時 method1u,
       nores = method1u_nores. 区間は data/edges_<tag から _nores を除いたもの>_<thr>.csv
出力: results/<tag>_bestfit_<thr>.npz
"""
import sys, os
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix, vstack, eye

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = sys.argv[1] if len(sys.argv) > 1 else '1eV'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'method1eq'
TAG = 'method1u_nores' if TAG == 'nores' else TAG
M1 = np.loadtxt(os.path.join(HERE, 'data', f'CRmat_{TAG}_{THR}_originalUnit.csv'), delimiter=',')
edges = np.loadtxt(os.path.join(HERE, 'data', f'edges_{TAG.replace("_nores", "")}_{THR}.csv'), delimiter=',')
n = M1.shape[1]

sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='scipy', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
conv = a.cm**2*a.sec
XS = 1e12/conv
data = np.asarray(a.Ratebin7, dtype=float)
Ms = M1*conv_mat
A = csr_matrix(Ms*XS/data[:, None])                 # 行を data で正規化
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
print(f'[{THR}] N_int = {n}:  M x = data feasible (chi2_min = 0)? {r0.status == 0}  ({r0.message})')
if r0.status != 0:
    sys.exit(1)
y = r0.x; tol = 1e-6*y[0]

def treads(v):
    b = [[0, 0]]
    for j in range(1, n):
        if abs(v[j]-v[j-1]) < tol: b[-1][1] = j
        else: b.append([j, j])
    return [tuple(t) for t in b]

blocks = treads(y); nv = len(blocks)-1; improved = True; nlp = 0
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
x = rf.x*XS
resid = np.max(np.abs(Ms@x/data - 1))
d = len(data)
print(f'  vertex steps {nv} -> merged steps {len(blocks)-1} (d-1 = {d-1});  '
      f'max|Mx/data-1| = {resid:.1e};  LP tests {nlp}')
xp = x*conv
for E in [0.3, 0.5, 1.0, 1.5, 1.9, 2.5]:
    if E < edges[0]: continue
    print(f'  Phi({E:.1f} MeV) = {xp[np.searchsorted(edges, E, side="right")-1]/1e12:.3f} x 1e12 cm^-2 s^-1')
np.savez(os.path.join(HERE, 'results', f'{TAG}_bestfit_{THR}.npz'), edges=edges, x=xp)
