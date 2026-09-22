"""段マージ LP テスト: chi2 最小を厳密に保ったまま隣接トレッドを併合できるか.
usage: python tread_merge_test.py <1eV|5eV> [NB]"""
import sys, os, numpy as np, cvxpy as cp

REPO='/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2'
THR = sys.argv[1] if len(sys.argv)>1 else '1eV'
NB  = sys.argv[2] if len(sys.argv)>2 else '180'
TAG = '' if NB=='180' else f'_n{NB}'
HERE=os.path.join(REPO,'notes','method1')
M1   = np.genfromtxt(os.path.join(HERE,'data',f'CRmat_method1_{THR}{TAG}_originalUnit.csv'), delimiter=',')
edges= np.genfromtxt(os.path.join(HERE,'data',f'edges_method1_{THR}{TAG}.csv'), delimiter=',')
n = len(edges)-1

sys.path.insert(0, os.path.join(REPO,THR)); os.chdir(os.path.join(REPO,THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='osqp', T=3.0)
cm_ = a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
a.CRmat=M1*cm_; a.n=n; a.M_matrix=a.c*a.CRmat
a.data_vector=a.c*a.Ratebin7; a.Bkg_vector=np.zeros(a.m)
a._build_ordering_constraint()
a._dmb_default,a._inv_d_default=a._make_dmb_inv(a.data_vector)
a._hess_default=a._build_hessian_from(a._dmb_default,a._inv_d_default)
a._baseline_result=None; a.set_solver('osqp'); a._backend.vertex_select=True
res=a.optimize(a.data_vector); conv=a.cm**2*a.sec
xbf=res.x.copy(); chi0=res.fun/a.c
Ms=a.M_matrix/a.c; mu=Ms@xbf

def treads(x, tol):
    b=[[0,0]]
    for j in range(1,n):
        if abs(x[j]-x[j-1])<tol: b[-1][1]=j
        else: b.append([j,j])
    return [tuple(t) for t in b]

tol=1e-6*xbf[0]
blocks=treads(xbf,tol)
w1_0=sum(1 for s0,e0 in blocks if e0==s0)
print(f"[{THR} NB={NB}] start: chi2/c={chi0:.2e}  treads={len(blocks)}  width-1={w1_0}")

# Delta chi2 <= 1e-3 (論文の縮退バンドと同じ閾値) を許すマージ判定
DCHI = 1e-3
d = a.data_vector
sqw = 1.0/np.sqrt(np.where(d>0, d, 1.0)*a.c)   # chi2/c の単位に正規化
CHI_BOUND = chi0 + DCHI
XS = float(xbf[0])                              # 変数スケール

def chi2_expr(y):                               # y = x/XS
    return cp.sum_squares(cp.multiply(sqw, d - (a.M_matrix*XS)@y))

def feasible2(bl):
    y=cp.Variable(n, nonneg=True)
    cons=[chi2_expr(y) <= CHI_BOUND, y[:-1]>=y[1:]]
    for s0,e0 in bl:
        if e0>s0: cons.append(y[s0:e0]==y[s0+1:e0+1])
    prob=cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
        return prob.status in ('optimal','optimal_inaccurate')
    except Exception:
        return False

ntest=0
improved=True
while improved:
    improved=False
    order=sorted(range(len(blocks)-1), key=lambda k: min(blocks[k][1]-blocks[k][0], blocks[k+1][1]-blocks[k+1][0]))
    for k in order:
        trial=blocks[:k]+[(blocks[k][0],blocks[k+1][1])]+blocks[k+2:]
        ntest+=1
        if feasible2(trial):
            blocks=trial; improved=True; break

# 最終代表元: マージ済み分割の等値制約の下で尾重み頂点則
y=cp.Variable(n, nonneg=True)
cons=[chi2_expr(y) <= CHI_BOUND, y[:-1]>=y[1:]]
for s0,e0 in blocks:
    if e0>s0: cons.append(y[s0:e0]==y[s0+1:e0+1])
prob=cp.Problem(cp.Minimize(chi2_expr(y)), cons)   # マージ済みパターン内で chi2 最小化
prob.solve(solver=cp.CLARABEL)
print(f"[final solve] status={prob.status}")
xm=(y.value*XS) if y.value is not None else xbf.copy()
chi_m=float(np.sum((a.data_vector - a.M_matrix@xm)**2/np.where(a.data_vector>0,a.data_vector,1)))/a.c
bl_m=treads(xm,tol)
w1_m=sum(1 for s0,e0 in bl_m if e0==s0)
print(f"[merge] LP tests: {ntest}   treads {len(blocks)} (realised {len(bl_m)})   width-1: {w1_0} -> {w1_m}")
print(f"[merge] chi2/c after = {chi_m:.2e} (before {chi0:.2e})   x[0] = {xm[0]*conv:.4e} (before {xbf[0]*conv:.4e})")
wid=[e0-s0+1 for s0,e0 in bl_m]
print(f"[merge] tread widths: min {min(wid)}  median {int(np.median(wid))}  max {max(wid)}")
np.savez(os.path.join(HERE,'results',f'merged_{THR}{TAG}.npz'), edges=edges, x_merged=xm*conv, x_orig=xbf*conv)
