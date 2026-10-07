"""Method 1 の「なぜ安定になったか」を切り分ける数値実験.

設定 (すべて 1eV, T=3):
  S_A  適応グリッド (n=272),  data = Ratebin7,  Bkg = 0        (現行 Method 1)
  S_B  一様グリッド  (n=272; 2-7 MeV を 92 等分), 同上          (グリッド仮説の対照)
  S_C  適応グリッド (n=272),  data = Ratebin7,  Bkg = RateDiff (h_i を引く: >2 MeV の
       列は真値 0 の冗長方向 -> 不安定化するかの検証)
  S_D  論文の 180 列 (2 MeV 打ち切り), data = Ratebin7, Bkg = RateDiff (Method 2 参照)

診断:
  * 列ノルムの分布 (max/min)
  * 重み付き行列 W^{1/2} M_s の特異値 (データ空間の条件数; ゼロ特異値 n-d 本は
    縮退面そのもので常に存在する)
  * OSQP 解 (面内部解) の chi2/c と滑らかさ (トレッド数), 頂点選択後のトレッド数
  * S_C: 2 MeV 超の成分が 0 に張り付くか, 2 MeV 以下の段構造が S_D と一致するか

trust-constr の真の再起動依存性 (osqp は決定論的で x0 を使わないため) は
  python stability_experiments.py tc_uniform   # S_B を trust-constr で (時間がかかる)
で別途実行する.

usage: python stability_experiments.py [osqp|tc_uniform]
"""
import sys, os, time
import numpy as np, pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else 'osqp'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THR = '1eV'; EMIN = 0.18; NB = 180
HERE = os.path.join(REPO, 'method1')

# ---------- 行列の用意 ----------
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
n_hi = len(edges_A) - 1 - NB                      # 適応区間の本数 (92)

edges_B = np.r_[edgesLow, np.linspace(2.0, 7.0, n_hi+1)[1:]]
M_B = build_matrix(edges_B)/s
M_B[:, :NB] = ref

# ---------- 解析クラス ----------
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
    # S_C の 2 MeV 以下は S_D (Method 2) と一致するか
    dev = np.max(np.abs(xC[:NB]-xD))/max(xD[0], 1e-300)
    print(f'\nS_C vs S_D below 2 MeV: max|dx|/x[0] = {dev:.2e}')

elif MODE == 'tc_uniform':
    # S_B を trust-constr で: 真の初期値依存性 (osqp は x0 を使わない)
    ao = make(M_B, len(edges_B)-1, 'none', 'osqp')
    scale = float(ao.optimize(ao.data_vector).x[0])   # 内部単位の典型スケール
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
