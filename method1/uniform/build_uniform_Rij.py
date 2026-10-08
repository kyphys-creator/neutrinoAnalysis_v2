"""Method 1 の応答行列を curlyR_i(E_nu) から一様区間で作り直す.

  R_ij = int_{E_j}^{E_{j+1}} curlyR_i(E_nu) dE_nu
  curlyR_i : Mathematica/<thr>/output/curlyR_table.csv (2000 点, cm^2/(MeV kg))
             の区分線形補間を厳密に積分 (CRmat180_from_curlyR と同じ方法)
  区間     : 2 MeV 以下と同じ幅 Delta = (2 - E_min)/180 の一様区間を 7 MeV まで
             (表が 7 MeV で終わるので, 7 MeV を超えない最大の本数 N)

検証:
  * 2 MeV 以下の 180 列が CRmat180_from_curlyR を再現するか (1 eV)
  * 閉包テスト: M x_true / Ratebin7 (x_true = 真の積分フラックス), 公開 CRmat180 との比較つき

usage: python build_uniform_Rij.py <1eV|5eV>
出力: data/CRmat_method1u_<thr>_originalUnit.csv, data/edges_method1u_<thr>.csv
"""
import sys, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR = sys.argv[1] if len(sys.argv) > 1 else '1eV'
EMIN = {'1eV': 0.18, '5eV': 0.41}[THR]
NB = 180

tab = pd.read_csv(os.path.join(REPO, 'Mathematica', THR, 'output', 'curlyR_table.csv'))
E = tab.iloc[:, 0].values
R = tab.iloc[:, 1:].values.T                       # (d, 2000)
d = R.shape[0]

def pl_integral(y, a, b):
    xs = np.r_[a, E[(E > a) & (E < b)], b]
    ys = np.interp(xs, E, y)
    return np.sum(0.5*(ys[1:]+ys[:-1])*np.diff(xs))

delta = (2.0 - EMIN)/NB
N = int(np.floor((E[-1] - EMIN)/delta + 1e-9))
edges = EMIN + delta*np.arange(N+1)
M = np.array([[pl_integral(R[i], edges[j], edges[j+1]) for j in range(N)] for i in range(d)])
print(f'[{THR}] d = {d}, Delta = {delta:.6f} MeV, N_int = {N} '
      f'({NB} below 2 MeV + {N-NB} above), last edge = {edges[-1]:.4f} MeV')

fc = os.path.join(REPO, 'Mathematica', THR, 'output', 'CRmat180_from_curlyR_originalUnit.csv')
if os.path.exists(fc):
    ref = np.loadtxt(fc, delimiter=',')
    print(f'  below 2 MeV vs CRmat180_from_curlyR: max rel dev '
          f'{np.max(np.abs(M[:, :NB]-ref))/ref.max():.1e}')

# 閉包テスト: 真の積分フラックスを区間下端で評価 (best fit 図と同じ規約)
sys.path.insert(0, os.path.join(REPO, THR)); os.chdir(os.path.join(REPO, THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='scipy', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
conv = a.cm**2*a.sec
Ef = np.linspace(EMIN, 7.2, 200001)
phif = np.interp(Ef, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'], right=0)
cum = np.r_[np.cumsum((0.5*(phif[1:]+phif[:-1])*np.diff(Ef))[::-1])[::-1], 0.0]

def Phi_at(e):                                     # 区間内平均の積分フラックス
    return np.interp(e, Ef, cum)

def closure(Mx, ed, data, Phi_off=0.0):
    mid = 0.5*(ed[1:]+ed[:-1])
    x = (Phi_at(mid) - Phi_off)/conv
    r = (Mx*conv_mat) @ x / data
    return r

r7 = closure(M, edges, a.Ratebin7)
print(f'  closure  M_curlyR(7 MeV) x_true / Ratebin7 : min {r7.min():.4f}  max {r7.max():.4f}')
pub = np.loadtxt(os.path.join(REPO, THR, 'CRmat', 'originalUnit', f'CRmat{NB}_originalUnit.csv'), delimiter=',')
e180 = np.linspace(EMIN, 2.0, NB+1)
r2c = closure(M[:, :NB], e180, a.Ratebin2, Phi_at(2.0))   # Ratebin2 は delta Phi = Phi - Phi(2 MeV)
r2p = closure(pub, e180, a.Ratebin2, Phi_at(2.0))
print(f'  closure  (<2 MeV) curlyR    x_true / Ratebin2 : min {r2c.min():.4f}  max {r2c.max():.4f}')
print(f'  closure  (<2 MeV) published x_true / Ratebin2 : min {r2p.min():.4f}  max {r2p.max():.4f}')

np.savetxt(os.path.join(HERE, 'data', f'CRmat_method1u_{THR}_originalUnit.csv'), M, delimiter=',')
np.savetxt(os.path.join(HERE, 'data', f'edges_method1u_{THR}.csv'), edges, delimiter=',')
print(f'  saved data/CRmat_method1u_{THR}_originalUnit.csv, data/edges_method1u_{THR}.csv')
