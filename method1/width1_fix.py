"""幅1トレッドだけを隣に併合する局所修正. 他の段構造は不変.
usage: python width1_fix.py <1eV|5eV>"""
import sys, os, numpy as np, cvxpy as cp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy import integrate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THR = sys.argv[1] if len(sys.argv)>1 else '1eV'
HERE=os.path.join(REPO,'method1'); RES=os.path.join(HERE,'results')
M1   = np.genfromtxt(os.path.join(HERE,'data',f'CRmat_method1_{THR}_originalUnit.csv'), delimiter=',')
edges= np.genfromtxt(os.path.join(HERE,'data',f'edges_method1_{THR}.csv'), delimiter=',')
n = len(edges)-1

sys.path.insert(0, os.path.join(REPO,THR)); os.chdir(os.path.join(REPO,THR))
import neutrino_analysis_band as nab
a = nab.NeutrinoAnalysis(background_scenario='none', intervals='180', GeV=0.32e16, solver='osqp', T=3.0)
cm_=a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
a.CRmat=M1*cm_; a.n=n; a.M_matrix=a.c*a.CRmat
a.data_vector=a.c*a.Ratebin7; a.Bkg_vector=np.zeros(a.m)
a._build_ordering_constraint()
a._dmb_default,a._inv_d_default=a._make_dmb_inv(a.data_vector)
a._hess_default=a._build_hessian_from(a._dmb_default,a._inv_d_default)
a._baseline_result=None; a.set_solver('osqp'); a._backend.vertex_select=True
res=a.optimize(a.data_vector); conv=a.cm**2*a.sec
xbf=res.x.copy(); chi0=res.fun/a.c

def treads(x, tol):
    b=[[0,0]]
    for j in range(1,n):
        if abs(x[j]-x[j-1])<tol: b[-1][1]=j
        else: b.append([j,j])
    return [tuple(t) for t in b]

tol=1e-6*xbf[0]
blocks=treads(xbf,tol)
ones=[i for i,(s0,e0) in enumerate(blocks) if e0==s0 and xbf[s0]>tol]  # 値0の尾は除外
print(f"[{THR}] start: chi2/c={chi0:.2e}  treads={len(blocks)}  width-1(nonzero)={len(ones)}")

d=a.data_vector; sqw=1.0/np.sqrt(np.where(d>0,d,1.0)*a.c)
XS=float(xbf[0]); A=a.M_matrix*XS
def solve_chi2(bl):
    y=cp.Variable(n, nonneg=True)
    cons=[y[:-1]>=y[1:]]
    for s0,e0 in bl:
        if e0>s0: cons.append(y[s0:e0]==y[s0+1:e0+1])
    prob=cp.Problem(cp.Minimize(cp.sum_squares(cp.multiply(sqw, d-A@y))), cons)
    prob.solve(solver=cp.CLARABEL)
    return (prob.value, y.value) if y.value is not None else (np.inf, None)

def merge_at(bl, i, side):
    if side=='L' and i>0:            return bl[:i-1]+[(bl[i-1][0],bl[i][1])]+bl[i+1:], True
    if side=='R' and i<len(bl)-1:    return bl[:i]+[(bl[i][0],bl[i+1][1])]+bl[i+2:], True
    return bl, False

# 幅1トレッドを順に処理 (インデックスは都度再計算)
cur=blocks
while True:
    tl=[(i,(s0,e0)) for i,(s0,e0) in enumerate(cur) if e0==s0 and xbf[min(s0,n-1)]>tol]
    # xbf でなく現行値で: 単純に値>0 の幅1を対象
    tl=[(i,(s0,e0)) for i,(s0,e0) in enumerate(cur) if e0==s0]
    tl=[(i,se) for i,se in tl if True]
    cand=None
    for i,(s0,e0) in tl:
        best=None
        for side in ('L','R'):
            bl2, ok = merge_at(cur, i, side)
            if not ok: continue
            c2,_=solve_chi2(bl2)
            if best is None or c2<best[0]: best=(c2, bl2, side)
        if best is not None:
            cand=best; break     # 一個ずつ確定
    if cand is None: break
    cur=cand[1]
chi_f, ym = solve_chi2(cur)
xm=ym*XS
bl_f=treads(xm,tol)
w1=sum(1 for s0,e0 in bl_f if e0==s0)
print(f"[fix] treads {len(blocks)} -> {len(cur)} (realised {len(bl_f)})   width-1 -> {w1}")
print(f"[fix] Delta chi2/c = {chi_f-chi0:.2e}   x[0]: {xbf[0]*conv:.4e} -> {xm[0]*conv:.4e}")
wid=[e0-s0+1 for s0,e0 in bl_f]
print(f"[fix] widths: min {min(wid)}  median {int(np.median(wid))}  max {max(wid)}")
np.savez(os.path.join(RES,f'merged_{THR}_w1.npz'), edges=edges, x_merged=xm*conv, x_orig=xbf*conv)

# ---- 図 ----
xg=np.logspace(-2,np.log10(7.0),4000)
ys=np.interp(xg,a.fig1Solid['MeV'],a.fig1Solid['cm**-2sec-1MeV-1'])
yd=np.interp(xg,a.fig1dashed['MeV'],a.fig1dashed['cm**-2sec-1MeV-1'])
Phi_s=np.array([integrate.trapezoid(ys[i:],xg[i:]) for i in range(len(xg))])
Phi_d=np.array([integrate.trapezoid(yd[i:],xg[i:]) for i in range(len(xg))])
norm=1e12
plt.style.use(os.path.join(REPO,'1eV','physrev.mplstyle'))
plt.figure(figsize=(8,6))
plt.plot(xg,Phi_s/norm,color='black',lw=3,label='With NC')
plt.plot(xg,Phi_d/norm,color='black',lw=3,ls='dashed',label='Without NC')
plt.scatter(edges[:-1],xm*conv/norm/conv*conv/norm*0+xm*conv/norm if False else xm*conv/norm, s=10, marker='o', color='#0072B2', zorder=5, label='No Bkg Best-fit')
plt.xscale('log'); plt.xlim(1e-1,7.3)
plt.xlabel(r"$E_\nu$ [MeV]", fontsize=30)
plt.ylabel(nab._phi_ylabel(norm).replace('<~2~','<~7~'), fontsize=30)
plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0,subs=np.arange(1.,10)*0.1,numticks=20))
plt.tick_params(axis='both',which='major',labelsize=23)
plt.tick_params(axis='both',which='minor',labelsize=23)
plt.legend(loc='upper right',fontsize=15,frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES,f'method1_bestfit_w1fix_{THR}.pdf'))
plt.savefig(os.path.join(RES,f'method1_bestfit_w1fix_{THR}.png'),dpi=115)
print('saved',f'method1_bestfit_w1fix_{THR}.pdf')
