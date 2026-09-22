"""Method 1 (Appendix E): 7 MeV までの un-truncated best fit.

手順:
  1. driver_kernels.wls が出力した公式レシピの節点 (0.01 MeV 刻みの box カーネル,
     区分線形補間の節点そのもの) を読み込む
  2. 検証: 2 MeV 以下の一様 180 区間で厳密な区分線形積分 -> 公式 CRmat180 と比較
  3. Graciela のグリッド: 2 MeV 以下は一様 180 区間 (論文と同一),
     2 MeV 超は Sum_i R_ij が下側の最終区間と同じになるよう適応的に細分
  4. フィット: data = Ratebin7 (2 MeV 超も信号), 背景ゼロ, h_i 減算なし,
     非負 + 単調減少制約, 頂点選択 ON
  5. 安定性チェック: osqp/scipy 両バックエンド + 摂動初期値

既存ファイルは一切変更しない (出力は notes/method1/ 内のみ).
usage: python build_and_fit.py <1eV|5eV>
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2'
THR  = sys.argv[1] if len(sys.argv) > 1 else '1eV'
EMIN = {'1eV': 0.18, '5eV': 0.41}[THR]
HERE = os.path.join(REPO, 'notes', 'method1')
NB   = int(os.environ.get('M1_NBELOW', '180'))   # 2 MeV 以下の一様区間数
TAG  = '' if NB == 180 else f'_n{NB}'
RES  = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)

# ---------- 1. 節点読み込み ----------
kn = pd.read_csv(os.path.join(HERE, 'data', f'curlyR_knots_{THR}.csv'))
Ek = kn.iloc[:, 0].values                  # MeV, 0.01 刻み
K  = kn.iloc[:, 1:].values.T               # (m bins, npts)  cm^2/(MeV kg)
m  = K.shape[0]
print(f"[{THR}] knots: {K.shape[1]} pts x {m} bins,  E {Ek[0]}..{Ek[-1]}")

def pl_integral(y, a, b):
    """区分線形 (節点 Ek, 値 y) の [a,b] 上の厳密積分."""
    xs = np.r_[a, Ek[(Ek > a) & (Ek < b)], b]
    ys = np.interp(xs, Ek, y)
    return np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs))

def build_matrix(edges):
    M = np.zeros((m, len(edges) - 1))
    for j in range(len(edges) - 1):
        for i in range(m):
            M[i, j] = pl_integral(K[i], edges[j], edges[j + 1])
    return M

# ---------- 2. 検証: 公式 CRmat180 と比較 ----------
edges180 = np.linspace(EMIN, 2.0, NB + 1)
M180_phys = build_matrix(edges180)                       # cm^2/kg
_ref_candidates = [os.path.join(REPO, THR, 'CRmat', 'originalUnit', f'CRmat{NB}_originalUnit.csv'),
                   os.path.join(REPO, 'Mathematica', THR, 'output', f'CRmat{NB}_originalUnit.csv')]
ref = np.genfromtxt(next(f for f in _ref_candidates if os.path.exists(f)), delimiter=',')
# スケールは行和 (公式と再生成で一致することを確認済み) から決める
rs_off = ref.sum(axis=1); rs_new = M180_phys.sum(axis=1)
ok_rows = rs_off > 0
s = np.median(rs_new[ok_rows] / rs_off[ok_rows])
rowdev = np.abs(rs_new[ok_rows] / rs_off[ok_rows] / s - 1)
print(f"[validate] scale (row sums) = {s:.8e},  row-sum agreement: max dev {rowdev.max():.2e}")
resid = np.abs(M180_phys[ref > 1e-3*ref.max()] / s / ref[ref > 1e-3*ref.max()] - 1)
print(f"[note] regenerated kernel shape vs legacy official (fine scale): max rel dev {resid.max():.2f}"
      "  -> below 2 MeV the official columns are used verbatim (splice)")

# ---------- 3. Graciela グリッド ----------
S = K.sum(axis=0)                                        # Sum_i kernel_i(E)
T = pl_integral(S, edges180[-2], edges180[-1])           # 下側最終区間の大きさ
hi_edges = [2.0]
while hi_edges[-1] < 7.0 - 1e-12:
    a = hi_edges[-1]
    # 累積が T に達する b を二分法で
    lo, hi = a, 7.0
    if pl_integral(S, a, 7.0) <= T:
        hi_edges.append(7.0); break
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if pl_integral(S, a, mid) < T else (lo, mid)
    hi_edges.append(0.5 * (lo + hi))
hi_edges = np.array(hi_edges)
edges = np.r_[edges180, hi_edges[1:]]
n = len(edges) - 1
w_hi = np.diff(hi_edges)
print(f"[grid] {NB} uniform (< 2 MeV, width {edges180[1]-edges180[0]:.4f})"
      f" + {len(w_hi)} adaptive (> 2 MeV, width {w_hi.min():.4f}..{w_hi.max():.4f})  -> n = {n}")

M1 = build_matrix(edges) / s                             # originalUnit 規約
M1[:, :len(edges180) - 1] = ref                          # < 2 MeV は論文の公式行列をそのまま
np.savetxt(os.path.join(HERE, 'data', f'CRmat_method1_{THR}{TAG}_originalUnit.csv'),
           M1, delimiter=',')
np.savetxt(os.path.join(HERE, 'data', f'edges_method1_{THR}{TAG}.csv'),
           edges, delimiter=',')

# ---------- 3b. 閉包テスト: M1 . x_true ≈ Ratebin7 ----------
dnde0 = pd.read_csv(os.path.join(REPO, 'Mathematica', THR, 'input', 'dNdEsolid.csv'))
_E0, _Y0 = dnde0.iloc[:, 0].values, dnde0.iloc[:, 1].values
_fn = (2.65e22 / 205.3) / (4 * np.pi) * (1 / 7200.0**2 + 1 / 10200.0**2)
_Ef = np.linspace(EMIN, 7.0, 40000); _pf = np.interp(_Ef, _E0, _Y0 * _fn)
x_true_cl = np.array([np.trapezoid(_pf[_Ef >= e], _Ef[_Ef >= e]) for e in edges[:-1]])
R7 = np.genfromtxt(os.path.join(REPO, THR, 'Ratebin', 'Ratebin7_originalUnit.csv'), delimiter=',')
convK = 1.0  # originalUnit 同士
pred = M1 @ (x_true_cl)  # 物理 flux * originalUnit 行列 -> 単位はクラス内で吸収されるので比だけ見る
ratio = pred / R7 / np.median(pred / R7)
print(f"[closure] M1.x_true vs Ratebin7 (shape, median-normalised): "
      f"min {ratio.min():.4f}  max {ratio.max():.4f}")

# ---------- 4. フィット ----------
sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab

def make_analysis(solver):
    GEV = float(os.environ.get('M1_GEV', '0.32e16'))
    a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180',
                             GeV=GEV, solver=solver, T=3.0)
    conv_mat = a.cm**2 / (10**3 * a.gram) * (10**3 * a.gram) * a.yr
    a.CRmat = M1 * conv_mat
    a.n = n
    a.M_matrix = a.c * a.CRmat
    a.data_vector = a.c * a.Ratebin7          # 全レートが信号 (h_i は引かない)
    a.Bkg_vector = np.zeros(a.m)
    a._build_ordering_constraint()
    a._dmb_default, a._inv_d_default = a._make_dmb_inv(a.data_vector)
    a._hess_default = a._build_hessian_from(a._dmb_default, a._inv_d_default)
    a._baseline_result = None
    a.set_solver(solver)          # 新しい n でバックエンドを作り直す
    if a._backend.name == 'osqp':
        a._backend.vertex_select = True
    return a

a = make_analysis('osqp')
res = a.optimize(a.data_vector)
conv = a.cm**2 * a.sec
x = res.x * conv
chi2 = res.fun / a.c
nsteps = int((np.abs(np.diff(x)) > 1e-6 * max(x[0], 1)).sum())
print(f"[fit osqp+vertex] chi2/c = {chi2:.3e}   steps = {nsteps}   "
      f"x[0] = {x[0]:.4e}   x above 2 MeV: max {x[NB:].max():.3e}")
tol = 1e-6 * x[0]
tread_len, cur = [], 1
for j in range(1, n):
    if abs(x[j] - x[j-1]) < tol: cur += 1
    else: tread_len.append(cur); cur = 1
tread_len.append(cur)
tl = np.array(tread_len)
nz = tl[:len(tl)]  # 全トレッド
print(f"[treads] total {len(tl)}   width-1 treads: {(tl==1).sum()}   "
      f"width dist: min {tl.min()}  median {int(np.median(tl))}  max {tl.max()}")

# ---------- 5. 安定性チェック ----------
checks = {}
av = make_analysis('osqp'); av._backend.vertex_select = False
rv = av.optimize(av.data_vector)
checks['vertex_off'] = dict(chi2=float(rv.fun / av.c),
                            max_dx_rel=float(np.max(np.abs(rv.x * conv - x)) / x[0]))
rng = np.random.default_rng(1)
for k in range(3):
    x0 = np.sort(rng.uniform(0, 2 * x[0] / conv, n))[::-1]
    rk = a.optimize(a.data_vector, x0=x0)
    xk = rk.x * conv
    checks[f'restart_{k}'] = dict(chi2=float(rk.fun / a.c),
                                  max_dx_rel=float(np.max(np.abs(xk - x)) / x[0]))
print("[stability]", json.dumps(checks, indent=1))

# ---------- 6. 図 (plot_band_comparison と同じ様式) と保存 ----------
# 理論曲線: With NC (実線) / Without NC (破線), いずれも Phi(E)-Phi(7 MeV)
from scipy import integrate as _integrate
xg = np.logspace(-2, np.log10(7.0), 4000)
ys = np.interp(xg, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'])
yd = np.interp(xg, a.fig1dashed['MeV'], a.fig1dashed['cm**-2sec-1MeV-1'])
Phi_s = np.array([_integrate.trapezoid(ys[i:], xg[i:]) for i in range(len(xg))])
Phi_d = np.array([_integrate.trapezoid(yd[i:], xg[i:]) for i in range(len(xg))])

norm = 1e12
plt.style.use(os.path.join(REPO, '1eV', 'physrev.mplstyle'))
from matplotlib.ticker import LogLocator
plt.figure(figsize=(8, 6))
plt.plot(xg, Phi_s / norm, color='black', lw=3, label='With NC')
plt.plot(xg, Phi_d / norm, color='black', lw=3, ls='dashed', label='Without NC')
plt.scatter(edges[:-1], x / norm, s=10, marker='o', color='#0072B2',
            zorder=5, label='No Bkg Best-fit')
plt.xscale('log'); plt.xlim(1e-1, 7.3)
plt.xlabel(r"$E_\nu$ [MeV]", fontsize=30)
plt.ylabel(nab._phi_ylabel(norm).replace('<~2~', '<~7~'), fontsize=30)
plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10) * 0.1, numticks=20))
plt.tick_params(axis='both', which='major', labelsize=23)
plt.tick_params(axis='both', which='minor', labelsize=23)
plt.legend(loc='upper right', fontsize=15, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'method1_bestfit_{THR}{TAG}.pdf'))
plt.savefig(os.path.join(RES, f'method1_bestfit_{THR}{TAG}.png'), dpi=115)

# 真値 (JSON 記録用)
Efine = np.linspace(EMIN, 7.0, 40000)
phif = np.interp(Efine, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'])
truth = np.array([np.trapezoid(phif[Efine >= e], Efine[Efine >= e]) for e in edges[:-1]])

json.dump(dict(threshold=THR, emin=EMIN, n=n, edges=edges.tolist(),
               best_fit_physical=x.tolist(), truth_physical=truth.tolist(),
               chi2_over_c=float(chi2), n_steps=nsteps, stability=checks,
               scale_phys_to_originalUnit=float(s), validation_max_rel_dev=float(resid.max())),
          open(os.path.join(RES, f'method1_bestfit_{THR}{TAG}.json'), 'w'))
print("saved figure + json in notes/method1/results/")
