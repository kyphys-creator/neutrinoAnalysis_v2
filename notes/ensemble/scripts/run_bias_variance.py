"""
Bias / variance / correlation of the fitted deltaPhi_j, exactly as Graciela described:

  1. take the data predicted by our best-fit function (Asimov, N_i = E(M x_bf + Bkg)),
  2. vary it directly with a Gaussian, N(N_i, sqrt(N_i)),
  3. refit each variation, keeping ALL nuisance parameters fixed (h_i, b_i at nominal),
  4. average and disperse the resulting deltaPhi_j.

"Truth" is the best fit to the real data, so this is a self-consistency test:
    bias_j = <deltaPhi_j> - deltaPhi_j^bf .

Run twice -- vertex_select ON (published staircase) and OFF (unique interior point) --
because the chi^2 minimiser is not unique and the ON case depends on which vertex the LP
returns.  Usage:  python run_bias_variance.py <thr:1eV|5eV> <bkg> <T> <M> <seed>
"""
import sys, os, json, time
import numpy as np, warnings
warnings.filterwarnings('ignore')

thr   = sys.argv[1] if len(sys.argv) > 1 else '5eV'
bkg   = sys.argv[2] if len(sys.argv) > 2 else 'flat'
T     = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
M     = int(sys.argv[4]) if len(sys.argv) > 4 else 500
seed  = int(sys.argv[5]) if len(sys.argv) > 5 else 20260821
modes = sys.argv[6].split(',') if len(sys.argv) > 6 else ['vertex_on','vertex_off']
NINT  = sys.argv[7] if len(sys.argv) > 7 else '180'
BKGVAR= (len(sys.argv) > 8 and sys.argv[8] in ('1','bkgvar','true'))

REPO = '/Users/koichiro/Documents/Claude/neutrinoAnalysis_v2'
os.chdir(os.path.join(REPO, thr))
sys.path.insert(0, os.path.join(REPO, thr))
import neutrino_analysis_band as nab

a = nab.NeutrinoAnalysis(background_scenario=bkg, intervals=NINT,
                         GeV=0.32e16, solver='osqp', T=T, bkg_penalty=BKGVAR)
rng_bkg = np.random.default_rng(seed + 7) if BKGVAR else None
conv = a.cm**2 * a.sec                       # natural -> cm^-2 s^-1
res  = a.optimize(a.data_vector)
x_bf = res.x.copy()
print(f"[{thr} {bkg} T={T}] free fit chi2/c={res.fun/a.c:.3e}  n={a.n} d={a.m}", flush=True)

out = {'threshold': thr, 'background': bkg, 'T': T, 'M': M, 'seed': seed,
       'n_int': int(a.n), 'd_bins': int(a.m), 'N_int_arg': NINT,
       'Ev_MeV': (0.5*(np.linspace(0.41 if thr=='5eV' else 0.18, 2.0, a.n+1)[:-1]
                       + np.linspace(0.41 if thr=='5eV' else 0.18, 2.0, a.n+1)[1:])).tolist(),
       'bkg_varied': bool(BKGVAR),
       'best_fit_physical': (x_bf*conv).tolist(), 'runs': {}}

# pseudo-data: Gaussian around the Asimov expectation of the best fit; h_i, b_i fixed
a.generate_pseudo_data(num_pseudo_data=M, seed=seed, x=x_bf)
pseudo = [p.copy() for p in a.pseudo_data_sets]

for tag, vsel in [t for t in (('vertex_on', True), ('vertex_off', False)) if t[0] in modes]:
    a._backend.vertex_select = vsel
    X, CHI2, nfail, t0 = [], [], 0, time.time()
    for k, pd in enumerate(pseudo):
        try:
            if BKGVAR:
                Bv_ev, _ = a._sample_varied_background(pd, rng_bkg)
                a._bkg_varied = Bv_ev * a.c / a.T
            r = a.optimize(pd*a.c/a.T, x0=x_bf.copy())
            if r.success or r.x is not None:
                X.append(r.x.copy()); CHI2.append(float(r.fun)/a.c)
            else: nfail += 1
        except Exception:
            nfail += 1
        finally:
            a._bkg_varied = None
        if (k+1) % 100 == 0:
            print(f"  [{tag}] {k+1}/{M}  {(time.time()-t0)/(k+1):.2f}s/fit", flush=True)
    X = np.array(X)*conv                      # (Meff, n) physical
    Meff = len(X)
    mean = X.mean(0); sd = X.std(0, ddof=1)
    truth = x_bf*conv
    bias = mean - truth
    mse  = ((X - truth)**2).mean(0)
    C    = np.cov(X, rowvar=False, ddof=1)
    sdo  = np.where(sd > 0, sd, np.nan)
    rho  = C/np.outer(sdo, sdo)
    q16,q50,q84 = np.percentile(X,[15.87,50,84.13],axis=0)
    out['runs'][tag] = {
        'M_eff': Meff, 'n_fail': nfail, 'seconds': time.time()-t0,
        'chi2': list(CHI2), 'chi2_bestfit_real_data': float(res.fun)/a.c,
        'median': q50.tolist(), 'q16': q16.tolist(), 'q84': q84.tolist(),
        'frac_above_truth': (X > truth).mean(0).tolist(),
        'samples': X.astype(float).tolist(),
        'mean': mean.tolist(), 'sd': sd.tolist(), 'bias': bias.tolist(),
        'mse': mse.tolist(),
        'mc_err_bias': (sd/np.sqrt(max(Meff,1))).tolist(),
        'corr': np.nan_to_num(rho, nan=0.0).tolist(),
    }
    print(f"[{tag}] Meff={Meff} fail={nfail} in {time.time()-t0:.0f}s", flush=True)
    for j in [0, 2, 8, 20, 45, 90, 135, 167, 179]:
        if j >= a.n: continue
        b, s, t = bias[j], sd[j], truth[j]
        print(f"   j={j:3d} true={t:.4e} mean={mean[j]:.4e} bias={b:+.3e}"
              f" ({100*b/t if t>0 else float('nan'):+.1f}%) sd={s:.3e}"
              f" b/sd={b/s if s>0 else float('nan'):+.2f}"
              f" +-{s/np.sqrt(max(Meff,1)):.2e}(MC)", flush=True)

outdir = os.path.join(REPO, 'notes/ensemble', f'data/N{NINT}')
os.makedirs(outdir, exist_ok=True)
fn = os.path.join(outdir, f'bias_variance_{thr}_{bkg}_T{T:g}_M{M}_N{NINT}_{"-".join(modes)}{"_bkgvar" if BKGVAR else ""}.json')
json.dump(out, open(fn, 'w'))
print("saved", fn, flush=True)
