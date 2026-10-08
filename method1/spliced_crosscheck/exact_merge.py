"""Perform only the merges allowed at Delta chi2 = 0 (Mx=mu exactly) -> exact representative with the fewest steps."""
import sys, os, numpy as np, cvxpy as cp
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THR=sys.argv[1]
M1=np.genfromtxt(os.path.join(HERE,'data',f'CRmat_method1_{THR}_originalUnit.csv'),delimiter=',')
edges=np.genfromtxt(os.path.join(HERE,'data',f'edges_method1_{THR}.csv'),delimiter=',')
n=len(edges)-1
sys.path.insert(0,os.path.join(REPO,THR)); os.chdir(os.path.join(REPO,THR))
import neutrino_analysis_band as nab
a=nab.NeutrinoAnalysis(background_scenario='none',intervals='180',GeV=0.32e16,solver='osqp',T=3.0)
cm_=a.cm**2/(10**3*a.gram)*(10**3*a.gram)*a.yr
a.CRmat=M1*cm_; a.n=n; a.M_matrix=a.c*a.CRmat
a.data_vector=a.c*a.Ratebin7; a.Bkg_vector=np.zeros(a.m)
a._build_ordering_constraint()
a._dmb_default,a._inv_d_default=a._make_dmb_inv(a.data_vector)
a._hess_default=a._build_hessian_from(a._dmb_default,a._inv_d_default)
a._baseline_result=None; a.set_solver('osqp'); a._backend.vertex_select=True
res=a.optimize(a.data_vector); conv=a.cm**2*a.sec
xbf=res.x.copy(); Ms=a.M_matrix/a.c; mu=Ms@xbf
tol=1e-6*xbf[0]
def treads(x):
    b=[[0,0]]
    for j in range(1,n):
        if abs(x[j]-x[j-1])<tol: b[-1][1]=j
        else: b.append([j,j])
    return [tuple(t) for t in b]
blocks=treads(xbf)
XS=float(xbf[0])
def feasible(bl):
    y=cp.Variable(n, nonneg=True)
    cons=[(Ms*XS)@y==mu, y[:-1]>=y[1:]]
    for s0,e0 in bl:
        if e0>s0: cons.append(y[s0:e0]==y[s0+1:e0+1])
    p=cp.Problem(cp.Minimize(0),cons)
    try:
        p.solve(solver=cp.HIGHS, time_limit=15.0)
        return p.status in ('optimal','optimal_inaccurate')
    except Exception: return False
improved=True; ntest=0
while improved:
    improved=False
    order=sorted(range(len(blocks)-1), key=lambda k: min(blocks[k][1]-blocks[k][0], blocks[k+1][1]-blocks[k+1][0]))
    for k in order:
        trial=blocks[:k]+[(blocks[k][0],blocks[k+1][1])]+blocks[k+2:]
        ntest+=1
        if feasible(trial): blocks=trial; improved=True; break
d=a.m
print(f"[{THR}] d={d}: vertex treads {len(treads(xbf))} (downward steps {len(treads(xbf))-1});"
      f"  exact-merge -> treads {len(blocks)} (steps {len(blocks)-1});  d-1 = {d-1};  LP tests {ntest}")

# final representative (tail-weighted vertex rule within the final partition, Mx=mu exactly)
y=cp.Variable(n, nonneg=True)
cons=[(Ms*XS)@y==mu, y[:-1]>=y[1:]]
for s0,e0 in blocks:
    if e0>s0: cons.append(y[s0:e0]==y[s0+1:e0+1])
tw=1000.0**(np.arange(n)/max(n-1,1))
cp.Problem(cp.Minimize(tw@y),cons).solve(solver=cp.HIGHS, time_limit=30.0)
xm=y.value*XS
chi_m=float(np.sum((a.data_vector-a.M_matrix@xm)**2/np.where(a.data_vector>0,a.data_vector,1)))/a.c
print(f"[final] chi2/c = {chi_m:.2e}  treads realised {len(treads(xm))}")
RES=os.path.join(HERE,'results')
np.savez(os.path.join(RES,f'exactmerge_{THR}.npz'), edges=edges, x=xm*conv, x_vertex=xbf*conv)

# figure (band style, blue points only)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy import integrate as _int
xg=np.logspace(-2,np.log10(7.0),4000)
ys_=np.interp(xg,a.fig1Solid['MeV'],a.fig1Solid['cm**-2sec-1MeV-1'])
yd_=np.interp(xg,a.fig1dashed['MeV'],a.fig1dashed['cm**-2sec-1MeV-1'])
Ps=np.array([_int.trapezoid(ys_[i:],xg[i:]) for i in range(len(xg))])
Pd=np.array([_int.trapezoid(yd_[i:],xg[i:]) for i in range(len(xg))])
norm=1e12
plt.style.use(os.path.join(REPO,'1eV','physrev.mplstyle'))
plt.figure(figsize=(8,6))
plt.plot(xg,Ps/norm,color='black',lw=3,label='With NC')
plt.plot(xg,Pd/norm,color='black',lw=3,ls='dashed',label='Without NC')
plt.scatter(edges[:-1],xm*conv/norm,s=10,marker='o',color='#0072B2',zorder=5,label='No Bkg Best-fit')
plt.xscale('log'); plt.xlim(1e-1,7.3)
plt.xlabel(r"$E_\nu$ [MeV]", fontsize=30)
plt.ylabel(nab._phi_ylabel(norm).replace('<~2~','<~7~'), fontsize=30)
plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0,subs=np.arange(1.,10)*0.1,numticks=20))
plt.tick_params(axis='both',which='major',labelsize=23)
plt.tick_params(axis='both',which='minor',labelsize=23)
plt.legend(loc='upper right',fontsize=15,frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RES,f'method1_bestfit_dminus1_{THR}.pdf'))
plt.savefig(os.path.join(RES,f'method1_bestfit_dminus1_{THR}.png'),dpi=115)
print('saved',f'method1_bestfit_dminus1_{THR}.pdf')
