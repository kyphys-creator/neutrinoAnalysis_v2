"""1 eV の 1-3 eV ビンだけ閉包が -3% ずれる原因の検証.

仮説: curlyR_i (CurlRboxG) は +-1 eV の箱型分解能を含むので, 1-3 eV ビンの応答は
E_nu^min = 0.18 MeV (分解能を無視して決めた下限) より下にも伸びている. 行列は
0.18 MeV で切るのでその分が欠ける. 一方データ Ratebin7 は分解能なし (legacy
5_Calculation.nb で ExprateResolution 版はコメントアウト, Exprate 版が有効).

curlyR_i(E) = C_i/E^3 * int E_R F^2 P_i(E_R) dE_R  (04_response_defs.wl の CurlRboxG)
  P_i 箱型: IntGboxmu(E_R, e1, e2, 1 eV),  分解能なし: 1[e1 <= E_R <= e2]
ここでは F = 1 (E_R <= 120 eV で F^2 > 0.998). 規格化 C_i は表と高エネルギー側で合わせる
(箱型は線形関数 E_R の積分を保存するので, E_R^max >> e2 では両者は厳密に一致).

usage: python resolution_cut_check.py
出力: data/CRmat_method1u_nores_1eV_originalUnit.csv (分解能なしカーネル, 一様区間)
"""
import os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(REPO, 'method1')
Mn = (0.93149410372*72 - 0.0725)*1e9                     # eV (01_setup.wl)
SIG = 1.0
BINS = np.array([1., 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
                 43, 45, 47, 49, 51, 56, 61, 66, 71, 81, 120])

tab = pd.read_csv(os.path.join(REPO, 'Mathematica', '1eV', 'output', 'curlyR_table.csv'))
Et = tab.iloc[:, 0].values; Rt = tab.iloc[:, 1:].values.T

def ERmax(E_MeV):
    E = E_MeV*1e6
    return 2*E**2/(Mn + 2*E)

ERg = np.r_[np.geomspace(1e-4, 0.999, 3000), np.linspace(1.0, 130.0, 52000)]
def box(er, e1, e2):
    g = lambda u: u*(u > 0)
    return ((g(e2-er+SIG) - g(e2-er-SIG)) - (g(e1-er+SIG) - g(e1-er-SIG)))/(2*SIG)

def kernel_box(E_MeV, i):                                # 形のみ
    e1, e2 = BINS[i], BINS[i+1]
    w = ERg*box(ERg, e1, e2)
    cw = np.r_[0.0, np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(ERg))]
    return np.interp(ERmax(E_MeV), ERg, cw)/E_MeV**3

def kernel_nores(E_MeV, i):                              # 解析的: int_{e1}^{min(e2,ERmax)} E_R dE_R
    e1, e2 = BINS[i], BINS[i+1]
    top = np.clip(ERmax(E_MeV), e1, e2)
    return 0.5*(top**2 - e1**2)/E_MeV**3

d = len(BINS)-1
hi = Et >= 3.0
C = np.array([np.median(Rt[i, hi]/kernel_box(Et[hi], i)) for i in range(d)])
dev = max(np.max(np.abs(C[i]*kernel_box(Et, i)/np.where(Rt[i] > 1e-12*Rt[i].max(), Rt[i], np.nan) - 1)[Rt[i] > 1e-3*Rt[i].max()])
          for i in range(d))
print(f'reconstructed box kernel vs curlyR_table on [0.18, 7] MeV: max rel dev {dev:.1e}')

# flux
sys.path.insert(0, os.path.join(REPO, '1eV')); os.chdir(os.path.join(REPO, '1eV'))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='scipy', T=3.0)
conv_mat = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr; conv = a.cm**2*a.sec
print(f'flux table starts at E = {a.fig1Solid["MeV"].min():.4f} MeV')
Ef = np.linspace(0.01, 7.2, 720001)
phif = np.interp(Ef, a.fig1Solid['MeV'], a.fig1Solid['cm**-2sec-1MeV-1'], left=0, right=0)
Phi = np.r_[np.cumsum((0.5*(phif[1:]+phif[:-1])*np.diff(Ef))[::-1])[::-1], 0.0]

lo = Ef < 0.18
for i in range(3):
    k = C[i]*kernel_box(Ef, i)
    miss = np.trapezoid((Phi*k)[lo], Ef[lo])/np.trapezoid((Phi*k)[~lo & (Ef <= 7.0)], Ef[~lo & (Ef <= 7.0)])
    print(f'bin {BINS[i]:.0f}-{BINS[i+1]:.0f} eV: response below 0.18 MeV (cut by the matrix) = {miss:.2%} of the rest')

# 分解能なしカーネルで一様区間の行列を作り, 閉包を比べる
edges = np.loadtxt(os.path.join(HERE, 'data', 'edges_method1u_1eV.csv'), delimiter=',')
Mbox = np.loadtxt(os.path.join(HERE, 'data', 'CRmat_method1u_1eV_originalUnit.csv'), delimiter=',')
Efine = np.linspace(edges[0], edges[-1], 200*(len(edges)-1)+1)
Knr = np.array([C[i]*kernel_nores(Efine, i) for i in range(d)])
cK = np.concatenate([np.zeros((d, 1)), np.cumsum(0.5*(Knr[:, 1:]+Knr[:, :-1])*np.diff(Efine), axis=1)], axis=1)
Mnr = np.diff(np.array([np.interp(edges, Efine, cK[i]) for i in range(d)]), axis=1)

m = Ef <= 7.0
Icum = np.r_[0.0, np.cumsum(0.5*(Phi[1:]+Phi[:-1])*np.diff(Ef))]
xt = np.diff(np.interp(edges, Ef, Icum))/np.diff(edges)
for name, M in [('box (curlyR_table)', Mbox), ('no resolution     ', Mnr)]:
    r = (M*conv_mat) @ (xt/conv) / a.Ratebin7
    print(f'closure {name}: 1-3 eV {r[0]:.4f}  3-5 eV {r[1]:.4f}  5-7 eV {r[2]:.4f}  ...  '
          f'81-120 eV {r[-1]:.4f}   (bin1 - bin2 = {r[0]-r[1]:+.4f})')
np.savetxt(os.path.join(HERE, 'data', 'CRmat_method1u_nores_1eV_originalUnit.csv'), Mnr, delimiter=',')
print('saved data/CRmat_method1u_nores_1eV_originalUnit.csv')
