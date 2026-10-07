"""Method 1 の純 scipy パイプライン: フィット + 頂点化 + 厳密マージ.

cvxpy/osqp を使わない版:
  * フィット      -- 解析クラスの solver='scipy' (trust-constr, 論文本体と同じ)
  * 頂点化 (crossover) -- scipy.optimize.linprog (HiGHS 単体法):
                    tail 重み最小化, M x = mu を厳密に保ち単調 polytope の頂点へ
  * 厳密マージ    -- Delta chi2 = 0 (M x = mu 厳密) の隣接トレッド併合を
                    linprog の実行可能性判定で貪欲に -> d-1 段
数値注意: 等式制約は y = x/XS (XS = x[0]) のスケール変数で課す. 生の M_s x = mu
だと係数 ~1e-7 に対し変数 ~1e8 で, HiGHS の絶対許容誤差がフィットを壊す
(neutrino_analysis_band._OSQPBackend._build_lp の注記と同じ理由).

usage: python fit_and_merge_scipy.py <1eV|5eV>
出力: results/exactmerge_<thr>.npz (edges, x, x_vertex) を上書き +
      旧 cvxpy 結果との比較を表示
"""
import sys, os, time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, vstack, eye

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THR = sys.argv[1] if len(sys.argv) > 1 else '1eV'
HERE = os.path.join(REPO, 'method1'); RES = os.path.join(HERE, 'results')

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
a.data_vector = a.c*a.Ratebin7          # 2 MeV 超も信号; h_i 減算なし
a.Bkg_vector = np.zeros(a.m)
a._build_ordering_constraint()
a._dmb_default, a._inv_d_default = a._make_dmb_inv(a.data_vector)
a._hess_default = a._build_hessian_from(a._dmb_default, a._inv_d_default)
a._baseline_result = None
a.set_solver('scipy')

t0 = time.time()
res = a.optimize(a.data_vector)
conv = a.cm**2*a.sec
x_int = res.x.copy()                    # trust-constr の面内部解
chi2 = lambda x: float(np.sum((a.data_vector - a.M_matrix@x)**2
                       / np.where(a.data_vector > 0, a.data_vector, 1)))/a.c
print(f'[{THR}] scipy trust-constr: {time.time()-t0:.1f}s  chi2/c = {chi2(x_int):.3e}  '
      f'x(Emin) = {x_int[0]*conv:.4e} cm^-2 s^-1')

# ---------- 頂点化 + マージ (すべて linprog / HiGHS) ----------
Ms = a.M_matrix/a.c
XS = float(x_int[0])
A = Ms*XS                               # 等式制約行列 (スケール済み)
mu = Ms @ x_int
b_eq = mu
tw = 1000.0**(np.arange(n)/max(n-1, 1)) # tail 重み: 高エネルギー端を 0 へ押す
# 単調性 y[j+1] <= y[j] : (n-1) x n の疎行列
Dmono = (eye(n, n, k=1) - eye(n, n)).tocsr()[:-1]

def solve_lp(blocks=None, feas_only=False):
    """blocks 内の隣接равные制約を追加した LP. feas_only なら c=0."""
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

r0 = solve_lp()                          # crossover: 同一 fit の頂点へ
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

rf = solve_lp(blocks)                    # 確定分割内の tail 重み代表元
assert rf.status == 0, rf.message
x_fit = rf.x*XS
d = a.m
print(f'[merge] d = {d}: steps -> {len(blocks)-1}  (d-1 = {d-1});  '
      f'chi2/c = {chi2(x_fit):.3e};  LP tests {nlp}')

# ---------- 旧 cvxpy 結果と比較して上書き保存 ----------
out = os.path.join(RES, f'exactmerge_{THR}.npz')
if os.path.exists(out):
    z = np.load(out)
    dev = np.max(np.abs(x_fit*conv - z['x']))/(z['x'][0])
    print(f'[compare] vs cvxpy pipeline: max|dx|/x[0] = {dev:.2e}')
np.savez(out, edges=edges, x=x_fit*conv, x_vertex=x_vertex*conv)
print(f'saved {out}')
