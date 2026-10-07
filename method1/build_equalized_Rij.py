"""Method 1: R_ij の大きさをそろえる適応グリッド (curlyR_i が大きいところで区間を細かく).

  T   = 幅 2 eV の E' ビン (大多数の行) の R_ij の最大値, 一様幅 Delta = (2 - E_min)/180 で計算
  区間 = E_min から 7 MeV へ順に, 各区間で max_i int curlyR_i <= T となる最大の幅
        (ただし Delta を超えない). curlyR_i が小さいところは一様幅 Delta のまま.
  行列 = 同じ区間で 2 通り:
        box   : curlyR_table.csv (+-1 eV 箱型分解能) の区分線形補間を厳密積分
        nores : 分解能なしカーネル C_i/E^3 int_{e1}^{min(e2,ERmax)} E_R dE_R
                (C_i は高エネルギー側で表に合わせる; resolution_cut_check.py と同じ)

usage: python build_equalized_Rij.py <1eV|5eV>
出力: data/edges_method1eq_<thr>.csv, data/CRmat_method1eq_<thr>_originalUnit.csv,
      data/CRmat_method1eq_nores_<thr>_originalUnit.csv
"""
import sys, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THR = sys.argv[1] if len(sys.argv) > 1 else '1eV'
EMIN = {'1eV': 0.18, '5eV': 0.41}[THR]
NB = 180
HERE = os.path.join(REPO, 'method1')
Mn = (0.93149410372*72 - 0.0725)*1e9                     # eV (01_setup.wl)
SIG = 1.0
BINS = np.array([1., 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
                 43, 45, 47, 49, 51, 56, 61, 66, 71, 81, 120])[{'1eV': 0, '5eV': 2}[THR]:]

tab = pd.read_csv(os.path.join(REPO, 'Mathematica', THR, 'output', 'curlyR_table.csv'))
Et = tab.iloc[:, 0].values; Rt = tab.iloc[:, 1:].values.T
d = Rt.shape[0]
assert d == len(BINS)-1

def pl_integral(y, a, b):
    xs = np.r_[a, Et[(Et > a) & (Et < b)], b]
    ys = np.interp(xs, Et, y)
    return np.sum(0.5*(ys[1:]+ys[:-1])*np.diff(xs))

# ---------- グリッド ----------
Ef = np.linspace(Et[0], Et[-1], 400001)
Rf = np.array([np.interp(Ef, Et, Rt[i]) for i in range(d)])
Cum = np.concatenate([np.zeros((d, 1)), np.cumsum(0.5*(Rf[:, 1:]+Rf[:, :-1])*np.diff(Ef), axis=1)], axis=1)
def colmax(a, b):
    return np.max([np.interp(b, Ef, Cum[i]) - np.interp(a, Ef, Cum[i]) for i in range(d)])

delta = (2.0 - EMIN)/NB
eU = EMIN + delta*np.arange(int(np.floor((Et[-1]-EMIN)/delta + 1e-9))+1)
w2 = np.diff(BINS) == 2
T = max(pl_integral(Rt[i], eU[j], eU[j+1]) for i in np.where(w2)[0] for j in range(len(eU)-1))

EMAX = Et[-1]
edges = [EMIN]
while edges[-1] < EMAX - 1e-12:
    a = edges[-1]
    b = min(a + delta, EMAX)
    if colmax(a, b) > T:
        lo, hi = a, b
        for _ in range(50):
            mid = 0.5*(lo+hi)
            (lo, hi) = (mid, hi) if colmax(a, mid) <= T else (lo, mid)
        b = lo
    edges.append(b)
if edges[-1] - edges[-2] < 0.5*delta and colmax(edges[-3], edges[-1]) <= T:
    del edges[-2]                                  # 7 MeV 端の短い余り区間は一つ前に併合
edges = np.array(edges)
w = np.diff(edges)
narrow = w < delta*(1-1e-6)
print(f'[{THR}] T = {T:.3e} cm^2/kg (max element of the 2-eV rows at width {delta:.6f} MeV)')
print(f'  N_int = {len(w)}  (uniform-width intervals {np.sum(~narrow)}, narrowed {narrow.sum()}); '
      f'last edge {edges[-1]:.4f} MeV')
print(f'  narrowing between {edges[:-1][narrow].min():.3f} and {edges[1:][narrow].max():.3f} MeV, '
      f'min width {w.min():.5f} MeV (= Delta/{delta/w.min():.1f})')

# ---------- 行列 (box) ----------
Mbox = np.array([[pl_integral(Rt[i], edges[j], edges[j+1]) for j in range(len(w))] for i in range(d)])
Mu = np.loadtxt(os.path.join(HERE, 'data', f'CRmat_method1u_{THR}_originalUnit.csv'), delimiter=',')
print(f'  box matrix: max element / T = {Mbox.max()/T:.3f}   (uniform grid: {Mu.max()/T:.2f})')

# ---------- 行列 (nores) ----------
def ERmax(E_MeV):
    E = E_MeV*1e6
    return 2*E**2/(Mn + 2*E)
ERg = np.r_[np.geomspace(1e-4, 0.999, 3000), np.linspace(1.0, 130.0, 52000)]
def box(er, e1, e2):
    g = lambda u: u*(u > 0)
    return ((g(e2-er+SIG) - g(e2-er-SIG)) - (g(e1-er+SIG) - g(e1-er-SIG)))/(2*SIG)
def kernel_box(E_MeV, i):
    wgt = ERg*box(ERg, BINS[i], BINS[i+1])
    cw = np.r_[0.0, np.cumsum(0.5*(wgt[1:]+wgt[:-1])*np.diff(ERg))]
    return np.interp(ERmax(E_MeV), ERg, cw)/E_MeV**3
def kernel_nores(E_MeV, i):
    e1, e2 = BINS[i], BINS[i+1]
    top = np.clip(ERmax(E_MeV), e1, e2)
    return 0.5*(top**2 - e1**2)/E_MeV**3
hiE = Et >= 3.0
C = np.array([np.median(Rt[i, hiE]/kernel_box(Et[hiE], i)) for i in range(d)])
Efine = np.unique(np.r_[np.linspace(edges[0], edges[-1], 400001), edges])
Knr = np.array([C[i]*kernel_nores(Efine, i) for i in range(d)])
cK = np.concatenate([np.zeros((d, 1)), np.cumsum(0.5*(Knr[:, 1:]+Knr[:, :-1])*np.diff(Efine), axis=1)], axis=1)
Mnr = np.diff(np.array([np.interp(edges, Efine, cK[i]) for i in range(d)]), axis=1)
print(f'  nores matrix: max element / T = {Mnr.max()/T:.3f}')

np.savetxt(os.path.join(HERE, 'data', f'edges_method1eq_{THR}.csv'), edges, delimiter=',')
np.savetxt(os.path.join(HERE, 'data', f'CRmat_method1eq_{THR}_originalUnit.csv'), Mbox, delimiter=',')
np.savetxt(os.path.join(HERE, 'data', f'CRmat_method1eq_nores_{THR}_originalUnit.csv'), Mnr, delimiter=',')
print(f'  saved data/edges_method1eq_{THR}.csv and both matrices')
