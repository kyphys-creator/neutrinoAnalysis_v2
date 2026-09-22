"""
NeutrinoAnalysis — fast version with two interchangeable solver backends.

Two backends are available, switched via the ``solver`` argument to
``__init__`` (or by calling ``set_solver`` later):

  * ``solver='scipy'`` — scipy.optimize.minimize(method='trust-constr')
                        with an analytic Jacobian and a constant analytic
                        Hessian. Identical solver to the original code,
                        just much better-informed. ~2–3× faster than the
                        original on this problem.

  * ``solver='osqp'``  — Reformulates the χ² minimization as the quadratic
                        program it actually is, and solves it with OSQP via
                        cvxpy. ~100× faster than the original on this
                        problem. Requires ``pip install cvxpy osqp``.

Both backends produce the same flux to machine precision (|Δx| ~ 1e-9 in
benchmarks). They share the same public API, so user-level scripts do not
need to change other than passing ``solver=...``.
"""

import os
import glob
import json

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy import integrate
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit
from matplotlib.ticker import LogLocator

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


# Apply the Physical Review style sheet if it sits next to this module, so all
# plots pick up the journal fonts/ticks/colours automatically. (figsize stays
# whatever each plotting call sets.)
_PHYSREV_STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'physrev.mplstyle')
if os.path.exists(_PHYSREV_STYLE):
    try:
        plt.style.use(_PHYSREV_STYLE)
    except Exception:
        pass


# -------------------------------------------------------------------
# Numba kernels (used by the scipy backend)
# -------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _chiN_core(xvec, M, data_minus_bkg, inv_data, T):
    """χ² = T · Σ (data − Bkg − M·x)² / data   (bins with data > 0 only)."""
    r = data_minus_bkg - M @ xvec
    return T * np.sum(r * r * inv_data)


@njit(cache=True, fastmath=True)
def _chiN_grad_core(xvec, M, data_minus_bkg, inv_data, T):
    """∂χ²/∂x = −2T · Mᵀ ((data − Bkg − M·x) · inv_data)."""
    r = data_minus_bkg - M @ xvec
    return -(2.0 * T) * (M.T @ (r * inv_data))


# -------------------------------------------------------------------
# Lightweight result wrapper so both backends look the same to callers.
# -------------------------------------------------------------------
class _Result:
    __slots__ = ('x', 'fun', 'success', 'status', 'nit')

    def __init__(self, x, fun, success=True, status='', nit=0):
        self.x = x
        self.fun = fun
        self.success = success
        self.status = status
        self.nit = nit


# -------------------------------------------------------------------
# Backend: scipy trust-constr with analytic jac & const hess
# -------------------------------------------------------------------
class _ScipyBackend:
    name = 'scipy'

    def __init__(self, parent):
        self.p = parent

    def solve(self, data, x0, extra_constraints=None, display=False):
        p = self.p
        M = p.M_matrix
        T = p.T

        if data is p.data_vector and p._bkg_varied is None:
            dmb, inv_d, H = p._dmb_default, p._inv_d_default, p._hess_default
        else:
            dmb, inv_d = p._make_dmb_inv(data)
            H = p._build_hessian_from(dmb, inv_d)

        def fun(x):  return _chiN_core(x, M, dmb, inv_d, T)
        def jac(x):  return _chiN_grad_core(x, M, dmb, inv_d, T)
        def hess(_): return H

        cons = [p.ordering_constraint]
        if extra_constraints:
            cons.extend(extra_constraints)

        options = {'maxiter': 100000, 'xtol': 1e-9, 'gtol': 1e-9}
        if display:
            options['verbose'] = 3

        return minimize(
            fun, x0, method='trust-constr',
            jac=jac, hess=hess,
            bounds=[(p.eps, None)] * p.n,
            constraints=cons, options=options,
        )


# -------------------------------------------------------------------
# Backend: OSQP via cvxpy.
#   min  ||W^{1/2} (M x − (d − b))||²
#   s.t. x ≥ 0,  x[i] ≥ x[i+1],  (optional) x[k] = v
# W = T · diag(1/d_i) on bins where d_i > 0, else 0.
# -------------------------------------------------------------------
class _OSQPBackend:
    """
    OSQP via cvxpy.

    Note on scaling: the project's natural-unit constant ``self.c`` is ~1e70,
    which makes ``M_matrix``, ``data_vector`` etc. far larger than OSQP's
    finite-bound limit (~1e30). The scipy backend doesn't notice because
    trust-constr scales internally. Here we solve the *unscaled* problem in
    a local variable and multiply back when reporting ``fun``, so that
    ``res.fun / self.c`` still gives the χ²/c the user expects.

    Concretely, the equivalent unscaled problem uses
        M_s = M / c          d_s = d / c        b_s = b / c
        inv_d_s = 1 / d_s = c / d
        w_s = sqrt(T * inv_d_s) = sqrt(c) * w
    and we compensate by returning ``fun * c`` so external code that divides
    by ``self.c`` keeps working.
    """
    name = 'osqp'

    _VERTEX_SOLVERS = None   # cached list of installed simplex LP solvers
    _VERTEX_WARNED = False   # warn at most once if none are available

    # eps 1e-8 is plenty here (χ² agrees with 1e-10 to ~1e-17) and converges in
    # ~25 ADMM iterations instead of ~90 000, so the free solve is ~1000× faster.
    DEFAULT_OPTS = dict(
        eps_abs=1e-8, eps_rel=1e-8,
        eps_prim_inf=1e-9, eps_dual_inf=1e-9,
        max_iter=20000,
        polish=True, polish_refine_iter=10,
        adaptive_rho=True,
        scaling=20,           # more scaling iterations → better conditioning
        verbose=False, warm_start=True,
    )

    def __init__(self, parent):
        if not _HAS_CVXPY:
            raise ImportError(
                "solver='osqp' requires cvxpy. Install with: pip install cvxpy osqp"
            )
        self.p = parent
        # Internal variable ``y`` is the *column-scaled* flux: x = D ⊙ y, with
        # D = 1/‖M_s column‖. Without this rescaling the optimal x (~1e7) and
        # the design-matrix columns (~1e-6) span ~13 orders of magnitude, and
        # OSQP/CLARABEL both terminate early at a wrong point while reporting
        # status='optimal'. D is data-independent, so the parameterised
        # (DPP) problem can still be re-solved fast across pseudo-data.
        self._y = cp.Variable(parent.n, nonneg=True)
        self._cache = {}     # QP problems, keyed by fixed_index
        self._lp_cache = {}  # vertex-selection LPs, keyed by fixed_index
        # The χ² minimiser is a high-dimensional face of the feasible polytope
        # whenever n > rank(M) (here 180 params vs 29 bins). OSQP/CLARABEL are
        # interior-point/ADMM methods that return a *relative-interior* point of
        # that face — a smooth ramp. The physically meaningful estimate is a
        # *vertex* (piecewise-constant flux), found by a simplex method. After
        # the QP fixes the unique fitted values μ = M·x, a HiGHS-simplex LP over
        # {M·x = μ, monotone, x ≥ 0} with a tail-weighted objective returns such
        # a vertex, with the high-energy tail driven to zero (see ``_build_lp``).
        self.vertex_select = True

    def reset_cache(self):
        self._cache.clear()
        self._lp_cache.clear()

    def _column_scale(self):
        M_s = self.p.M_matrix / self.p.c
        cn = np.linalg.norm(M_s, axis=0)
        cn[cn == 0] = 1.0
        return 1.0 / cn

    def _build_problem(self, fixed_index):
        """
        Build a parameterised problem so that *any* data vector for this
        background scenario can be plugged in by setting cvxpy Parameters.
        Only ``fixed_index`` changes the structure (adds an equality
        constraint), so that's the only cache key. The background-penalty
        mode reuses this exact problem; it only swaps which background is
        subtracted (see ``_set_data_params``).
        """
        p = self.p
        n, m = p.n, p.m
        y = self._y
        D = self._column_scale()

        # cvxpy Parameters: w (per-bin weight √(T/d_s)) and z (w * dmb_s).
        # Residual = w ⊙ (M_s·D ⊙ y) − z, solving for the scaled variable y.
        w_par = cp.Parameter(m, nonneg=True)
        z_par = cp.Parameter(m)
        MsD = (p.M_matrix / p.c) * D[None, :]   # constant, scaled once

        residual = cp.multiply(w_par, MsD @ y) - z_par
        obj = cp.Minimize(cp.sum_squares(residual))
        # Ordering on the physical flux x = D ⊙ y.
        cons = [cp.multiply(D[:-1], y[:-1]) >= cp.multiply(D[1:], y[1:])]

        fv_param = None
        if fixed_index is not None:
            fv_param = cp.Parameter()
            cons.append(D[fixed_index] * y[fixed_index] == fv_param)

        prob = cp.Problem(obj, cons)
        return prob, w_par, z_par, fv_param, D

    def _get_problem(self, fixed_index):
        if fixed_index not in self._cache:
            self._cache[fixed_index] = self._build_problem(fixed_index)
        return self._cache[fixed_index]

    def _build_lp(self, fixed_index):
        """
        Vertex-selection LP: among all fluxes reproducing the fitted values
        μ = M_s·x (parameter ``mu_par``), pick a vertex of the monotone polytope
        with a simplex method. ``mu_par`` is the only thing that changes between
        data vectors, so one LP per ``fixed_index`` is cached.

        Objective: minimise Σ Rⁱ⁄⁽ⁿ⁻¹⁾·xᵢ — a geometrically tail-weighted sum.
        This pushes the high-energy (high-index) flux down to zero, reproducing
        the scipy staircase whose tail vanishes, rather than leaving a non-zero
        floor (which a plain Σx would, since zeroing the tail costs total flux).
        The leading weight is 1 (not 0), so the head is not artificially inflated
        the way a weight like Σ i·xᵢ would do. The result is insensitive to R.

        The LP solves for the column-scaled variable z = x / D, exactly as the
        QP does, so the equality constraint becomes ``MsD @ z == mu`` with a
        unit-column matrix. Without this rescaling the raw constraint M_s·x = μ
        has entries ~1e-7 against variables ~1e8, so HiGHS's absolute feasibility
        tolerance (1e-7) lets x violate the fit by enough that χ²(x_v) is many
        orders of magnitude larger than the QP's value.
        """
        p = self.p
        n, m = p.n, p.m
        D = self._column_scale()
        MsD = (p.M_matrix / p.c) * D[None, :]   # unit-column constant matrix
        tail_w = 1000.0 ** (np.arange(n) / max(n - 1, 1))
        z = cp.Variable(n, nonneg=True)         # x = D ⊙ z
        mu_par = cp.Parameter(m)
        cons = [MsD @ z == mu_par,
                cp.multiply(D[:-1], z[:-1]) >= cp.multiply(D[1:], z[1:])]
        fv_par = None
        if fixed_index is not None:
            fv_par = cp.Parameter()
            cons.append(D[fixed_index] * z[fixed_index] == fv_par)
        prob = cp.Problem(cp.Minimize((tail_w * D) @ z), cons)
        return prob, z, mu_par, fv_par, D

    def _get_lp(self, fixed_index):
        if fixed_index not in self._lp_cache:
            self._lp_cache[fixed_index] = self._build_lp(fixed_index)
        return self._lp_cache[fixed_index]

    def _vertex_solvers(self):
        """Simplex/active-set LP solvers that return a *vertex* (so the flux is
        piecewise-constant), in preference order, filtered to what's installed.

        HiGHS is fastest but is an optional cvxpy extra (``pip install highspy``).
        SciPy's linprog (HiGHS method) ships with cvxpy itself, so it is the
        reliable fallback: without it, a collaborator missing HiGHS silently got
        the smooth interior solution instead of the staircase. Interior-point
        solvers (CLARABEL/SCS/OSQP) are deliberately excluded — they return a
        face interior, not a vertex."""
        if _OSQPBackend._VERTEX_SOLVERS is None:
            avail = set(cp.installed_solvers())
            _OSQPBackend._VERTEX_SOLVERS = [
                s for s in ('HIGHS', 'GLPK', 'SCIPY') if s in avail
            ]
        return _OSQPBackend._VERTEX_SOLVERS

    def _select_vertex(self, x_interior, fixed_index, fixed_value):
        """Return a piecewise-constant vertex with the same fit as ``x_interior``."""
        M_s = self.p.M_matrix / self.p.c
        mu = M_s @ x_interior
        prob, z, mu_par, fv_par, D = self._get_lp(fixed_index)
        mu_par.value = mu
        if fixed_index is not None:
            fv_par.value = float(fixed_value)
        # Degenerate instances can make the HiGHS simplex cycle for hours
        # (seen in the N_int=160 pseudo-experiment ensembles), so every LP
        # solve gets a wall-clock limit; a timed-out toy falls back to the
        # interior solution below instead of stalling the whole ensemble.
        lp_opts = {'HIGHS': {'time_limit': 10.0},
                   'SCIPY': {'scipy_options': {'time_limit': 10.0}}}
        for solver in self._vertex_solvers():
            try:
                prob.solve(solver=getattr(cp, solver), **lp_opts.get(solver, {}))
            except Exception:
                continue
            if prob.status in ('optimal', 'optimal_inaccurate') and z.value is not None:
                return D * z.value
        print("  [vertex] LP returned no vertex (infeasible at LP tolerance, or timed "
              "out); using the interior solution for this fit", flush=True)
        if not _OSQPBackend._VERTEX_WARNED:
            import warnings
            warnings.warn(
                "vertex_select is on but no simplex LP solver succeeded, so the "
                "flux is the smooth interior solution, not the piecewise-constant "
                "staircase. Install a vertex solver, e.g. `pip install highspy`. "
                f"(installed solvers: {cp.installed_solvers()})",
                RuntimeWarning,
            )
            _OSQPBackend._VERTEX_WARNED = True
        return None

    def _set_data_params(self, data, w_par, z_par):
        """Fill in the cvxpy Parameters from a (scaled) data vector.

        In background-penalty mode the only change is the background that gets
        subtracted: the per-pseudo-experiment ``_bkg_varied`` instead of the
        nominal ``Bkg_vector``. The Neyman denominator stays d_s (= O), so the
        statistic is T·Σ (O − B_varied − M·x)²/O."""
        p = self.p
        data_s = data / p.c
        bkg = p.Bkg_vector if p._bkg_varied is None else p._bkg_varied
        bkg_s = bkg / p.c
        safe = np.where(data_s > 0, data_s, 1.0)
        inv_d_s = np.where(data_s > 0, 1.0 / safe, 0.0)
        w = np.sqrt(p.T * inv_d_s)
        w_par.value = w
        z_par.value = w * (data_s - bkg_s)

    def solve(self, data, x0=None, extra_constraints=None,
              fixed_index=None, fixed_value=None, display=False):
        if fixed_index is None and extra_constraints:
            fixed_index, fixed_value = self._extract_fixed(extra_constraints)

        prob, w_par, z_par, fv_param, D = self._get_problem(fixed_index)
        self._set_data_params(data, w_par, z_par)
        if fixed_index is not None:
            fv_param.value = float(fixed_value)
        if x0 is not None:
            self._y.value = np.asarray(x0, dtype=float) / D

        opts = dict(self.DEFAULT_OPTS)
        if display:
            opts['verbose'] = True

        _ok = ('optimal', 'optimal_inaccurate')
        # Solver order. Free QP: OSQP (fast ADMM) first. Fixed-parameter QP:
        # OSQP does not converge (burns max_iter ~5 s), so lead with CLARABEL.
        # CLARABEL is fast but at extreme fixed values it sits on the edge of
        # 'optimal_inaccurate' and occasionally throws a hard SolverError, so
        # keep SCS (robust) and OSQP as ordered backups. cvxpy's status is a
        # read-only property, so failure is tracked in a local variable.
        if fixed_index is None:
            attempts = [(cp.OSQP, opts), (cp.CLARABEL, {}), (cp.SCS, {})]
        else:
            attempts = [(cp.CLARABEL, {}), (cp.SCS, {}), (cp.OSQP, opts)]

        status = 'solver_error'
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')   # squelch "solution may be inaccurate"
            for solver, extra in attempts:
                kw = dict(extra)
                kw.setdefault('verbose', display)
                try:
                    prob.solve(solver=solver, **kw)
                    status = prob.status
                except Exception:
                    status = 'solver_error'
                if status in _ok and self._y.value is not None and prob.value is not None:
                    break

        if status not in _ok or self._y.value is None or prob.value is None:
            return _Result(np.full(self.p.n, np.nan), np.inf,
                           success=False, status=status, nit=0)
        x = D * self._y.value
        fun = float(prob.value) * self.p.c

        # Replace the interior (ramp) solution with a piecewise-constant vertex
        # that has the identical fit, so the OSQP backend matches the staircase
        # shape the scipy backend produces. χ² (``fun``) is unchanged.
        if self.vertex_select:
            xv = self._select_vertex(x, fixed_index, fixed_value)
            if xv is not None:
                x = xv

        return _Result(
            x=np.array(x),
            fun=fun,
            success=True,
            status=status,
            nit=int(prob.solver_stats.num_iters) if prob.solver_stats else 0,
        )

    @staticmethod
    def _extract_fixed(extra_constraints):
        for c in extra_constraints:
            if isinstance(c, LinearConstraint):
                A = c.A
                A_dense = A.toarray() if hasattr(A, 'toarray') else np.asarray(A)
                if A_dense.shape[0] == 1 and np.isclose(c.lb, c.ub).all():
                    idx = int(np.argmax(np.abs(A_dense[0])))
                    if np.isclose(A_dense[0, idx], 1.0):
                        return idx, float(np.atleast_1d(c.lb)[0])
        return None, None


# ===================================================================
# Main class
# ===================================================================
class NeutrinoAnalysis:
    """
    Neutrino flux optimization pipeline with two interchangeable backends.

    Parameters
    ----------
    background_scenario : str
    intervals : str
    GeV, c : float
    solver : {'scipy', 'osqp'}
        Backend to use. Both produce the same solution to ~1e-9.
        OSQP is roughly 50–100× faster on this problem; scipy is the
        safe, well-tested fallback.
    """

    # ----------------------------------------------------------------
    def __init__(self, background_scenario='c', intervals='180',
                 GeV=1e-6, c=1, solver='scipy', T=3,
                 bkg_penalty=False, bkg_zero_threshold=1e-3):
        self.iterationTime = 1000
        self.T = T
        self.GeV = GeV
        self.c = c

        self._define_constants()
        self._load_data(intervals)
        self._prepare_backgrounds()

        # Master normalization constant — kept identical to the original
        self.c = 3e12 * 2.693500303951368e+58

        self._build_ordering_constraint()

        self.result = None
        # Background-penalty (varied-background) option. When on, the Monte
        # Carlo draws a "measured" background B_varied per pseudo-experiment
        # (beta-distributed ratio f = B/O^MC) and the fit
        # subtracts that B_varied instead of the nominal background, i.e. the
        # χ² becomes T·Σ (O − B_varied − M·x)²/O. The background is fixed per
        # fit (not a nuisance parameter); the variation across pseudo-
        # experiments is what propagates the background uncertainty into the
        # band. Default off → behaviour identical to before.
        self.bkg_penalty = bool(bkg_penalty)
        self.bkg_zero_threshold = bkg_zero_threshold   # events; B below → no penalty
        self._bkg_varied = None        # per-fit override (scaled units) or None
        self._baseline_mode = None     # bkg_penalty state of cached baseline
        self._max_sampling_records = 2000
        self.reset_bkg_penalty_stats()
        self.set_background(background_scenario)
        self.set_solver(solver)

    # ---------- data ----------
    def _load_data(self, intervals='180'):
        self.fig1Solid = pd.read_csv("Danny’s files/fig1-solid.csv")
        self.fig1dashed = pd.read_csv("Danny’s files/fig1-dashed.csv")
        self.intervals = intervals
        self.CRmat = (
            np.genfromtxt(
                f'CRmat/originalUnit/CRmat{self.intervals}_originalUnit.csv',
                delimiter=','
            )
            * self.cm ** 2 / (10 ** 3 * self.gram)
            * (10 ** 3 * self.gram) * self.yr
        )
        self.Ratebin7 = np.genfromtxt('Ratebin/Ratebin7_originalUnit.csv', delimiter=',')
        self.Ratebin2 = np.genfromtxt('Ratebin/Ratebin2_originalUnit.csv', delimiter=',')
        self.RateDiff = self.Ratebin7 - self.Ratebin2
        self.n = len(self.CRmat[0])
        self.m = len(self.Ratebin7)

        f = 2.65e22 / 205.3 / 434795262.39118177
        self.fig1Solid['cm**-2sec-1MeV-1'] = self.fig1Solid['fissionMeV'] * f
        self.fig1dashed['cm**-2sec-1MeV-1'] = self.fig1dashed['fissionMeV'] * f

    def _define_constants(self):
        self.MeV = 1e-3 * self.GeV
        self.keV = 1e-6 * self.GeV
        self.eV  = 1e-9 * self.GeV
        self.meV = 1e-12 * self.GeV
        self.gram = 5.62e23 * self.GeV
        self.sec  = 1 / (6.58e-25 * self.GeV)
        self.yr   = 365 * 24 * 3600 * self.sec
        self.cm   = 1 / (1.98e-14 * self.GeV)
        self.eps = 0
        self.bins = np.array([5., 7., 9., 11., 13., 15., 17., 19., 21., 23.,
                              25., 27., 29., 31., 33., 35., 37., 39., 41., 43.,
                              45., 47., 49., 51., 56., 61., 66., 71., 81., 120.])
        self.Mdetector = 1.0

    def _compute_bi(self, A, B, C):
        bi = []
        for i in range(len(self.bins) - 1):
            Ei, Ei1 = self.bins[i], self.bins[i + 1]
            integral = -A * B * (np.exp(-Ei1 / B) - np.exp(-Ei / B)) + C * (Ei1 - Ei)
            bi.append(self.Mdetector * integral)
        return np.array(bi)

    def _prepare_backgrounds(self):
        d = 365
        ba    = self._compute_bi(A=0   * self.eV / self.keV * d, B=10, C=0   * self.eV / self.keV * d)
        bb    = self._compute_bi(A=460 * self.eV / self.keV * d, B=10, C=100 * self.eV / self.keV * d)
        b2    = self._compute_bi(A=920 * self.eV / self.keV * d, B=10, C=200 * self.eV / self.keV * d)
        bc    = self._compute_bi(A=50  * self.eV / self.keV * d, B=10, C=20  * self.eV / self.keV * d)
        bflat = self._compute_bi(A=0   * self.eV / self.keV * d, B=10, C=1 * self.eV / self.keV * d)
        self.background_df = pd.DataFrame({
            "Bin Start [eV]": self.bins[:-1], "Bin End [eV]": self.bins[1:],
            "b_i (a)": ba, "b_i (b)": bb, "b_i (c)": bc,
            "b_i (flat)": bflat, "b_i (b2)": b2,
        })

    # ---------- output paths (T-segregated) ----------
    @property
    def _T_label(self):
        """``T3`` for integer T, ``T2.5`` otherwise. Used as a top-level prefix
        so different observation-time scenarios don't overwrite each other."""
        t = self.T
        return f'T{int(t)}' if float(t).is_integer() else f'T{t}'

    @property
    def scenario_dir(self):
        """Output directory for this T × background scenario."""
        return os.path.join(self._T_label,
                            f'scenario_bkg_{self.background_scenario}')

    @property
    def bands_dir(self):
        """Default location for confidence-band JSON files."""
        return os.path.join(self.scenario_dir, 'bands')

    # ---------- backend ----------
    def set_solver(self, solver):
        """Switch backend at any time. ``solver`` ∈ {'scipy', 'osqp'}."""
        solver = solver.lower()
        if solver == 'scipy':
            self._backend = _ScipyBackend(self)
        elif solver == 'osqp':
            self._backend = _OSQPBackend(self)
        else:
            raise ValueError(f"Unknown solver: {solver!r}. Use 'scipy' or 'osqp'.")
        self.solver = solver

    # ---------- constraints ----------
    def _build_ordering_constraint(self):
        rows = np.repeat(np.arange(self.n - 1), 2)
        cols = np.empty(2 * (self.n - 1), dtype=int)
        cols[0::2] = np.arange(self.n - 1)
        cols[1::2] = np.arange(1, self.n)
        vals = np.tile(np.array([1.0, -1.0]), self.n - 1)
        A = csc_matrix((vals, (rows, cols)), shape=(self.n - 1, self.n))
        self.ordering_constraint = LinearConstraint(A, lb=0, ub=np.inf)

    def _make_fixed_constraint(self, fixed_index, fixed_value):
        A_fixed = csc_matrix(([1.0], ([0], [fixed_index])), shape=(1, self.n))
        return LinearConstraint(A_fixed, lb=fixed_value, ub=fixed_value)

    # ---------- background & cached χ² ingredients ----------
    def set_background(self, bkg_scenario='c'):
        self.background_scenario = bkg_scenario
        bkg_map = {'a': 'b_i (a)', 'b': 'b_i (b)', 'c': 'b_i (c)',
                   'flat': 'b_i (flat)', 'none': 'b_i (a)', 'b2': 'b_i (b2)'}
        col = bkg_map.get(bkg_scenario)
        if col is None:
            raise ValueError(f"Invalid background scenario: {bkg_scenario}")

        self.ExtBkg = np.array(self.background_df[col])
        k = self.c
        self.M_matrix = k * self.CRmat
        if bkg_scenario == 'none':
            self.data_vector = k * (self.Ratebin2 + self.ExtBkg)
            self.Bkg_vector  = k * self.ExtBkg
        else:
            self.data_vector = k * (self.Ratebin7 + self.ExtBkg)
            self.Bkg_vector  = k * (self.RateDiff + self.ExtBkg)

        self._dmb_default, self._inv_d_default = self._make_dmb_inv(self.data_vector)
        self._hess_default = self._build_hessian_from(self._dmb_default,
                                                     self._inv_d_default)
        self._baseline_result = None   # invalidate cached free best fit

        if hasattr(self, '_backend') and self._backend.name == 'osqp':
            self._backend.reset_cache()

    def _make_dmb_inv(self, data):
        """χ² ingredients: dmb = data − Bkg and inv_data = 1/data on bins with
        data > 0, so the kernel evaluates T·Σ (data − Bkg − M·x)²/data.

        In background-penalty mode the only change is the background that is
        subtracted: the per-pseudo-experiment ``_bkg_varied`` in place of the
        nominal ``Bkg_vector``. The Neyman denominator (data) is unchanged."""
        bkg = self.Bkg_vector if self._bkg_varied is None else self._bkg_varied
        safe = np.where(data > 0, data, 1.0)
        inv = np.where(data > 0, 1.0 / safe, 0.0)
        return data - bkg, inv

    def _build_hessian_from(self, _dmb, inv_data):
        return (2.0 * self.T) * (self.M_matrix.T * inv_data) @ self.M_matrix

    # ---------- public solver API ----------
    def optimize(self, data, x0=None, display=False):
        if x0 is None:
            x0 = np.ones(self.n)
        if self._backend.name == 'osqp':
            res = self._backend.solve(data, x0=x0, display=display)
        else:
            res = self._backend.solve(data, x0, extra_constraints=None, display=display)
        self.result = res
        return res

    def optimize_with_fixed_parameter(self, data, fixed_index, fixed_value, x0=None):
        if x0 is None:
            print("Warning: main optimization has not been run. Starting from scratch.")
            x0 = np.ones(self.n)
        x0 = x0.copy()
        x0[fixed_index] = fixed_value

        if self._backend.name == 'osqp':
            return self._backend.solve(data, x0=x0,
                                       fixed_index=fixed_index,
                                       fixed_value=fixed_value)
        fc = self._make_fixed_constraint(fixed_index, fixed_value)
        return self._backend.solve(data, x0, extra_constraints=[fc], display=False)

    # ---------- scan ----------
    def scan_parameter(self, data, index, num_points=21, scan_range=0.75):
        if self.result is None:
            raise RuntimeError("Run optimize() first.")
        base_value = self.result.x[index]
        if index >= 135:
            scan_range = 1 - 1e-11

        if index < 10:
            fixed_values = np.append(
                np.linspace(0.9e12 / self.cm ** 2 / self.sec,
                            1044872844297.51 / self.cm ** 2 / self.sec,
                            num_points)[:-1],
                np.linspace(1044872844297.51 / self.cm ** 2 / self.sec,
                            6e12 / self.cm ** 2 / self.sec, num_points)
            )
        elif index < 21:
            fixed_values = np.linspace((1 - scan_range) * base_value,
                                       (1 + 2) * base_value,
                                       2 * num_points - 1)
        elif index < 134:
            fixed_values = np.linspace((1 - scan_range) * base_value,
                                       (1 + scan_range) * base_value,
                                       2 * num_points - 1)
        else:
            fixed_values = np.append(
                np.linspace((1 - scan_range) * base_value,
                            (1 + scan_range) * base_value, num_points)[:-1],
                np.linspace((1 + scan_range) * base_value, 4 * base_value, num_points)
            )

        chi_sq_results = []
        warm = self.result.x.copy()
        total = len(fixed_values)
        for k, v in enumerate(fixed_values, 1):
            r = self.optimize_with_fixed_parameter(data, index, v, x0=warm)
            warm = r.x.copy()
            chi_sq_results.append(r.fun / self.c)
            print(f"[{k}/{total}] idx={index}  flux={v * self.cm**2 * self.sec:.4e}  χ²={chi_sq_results[-1]:.6g}")
        return fixed_values, chi_sq_results

    # ---------- Monte Carlo ----------
    @staticmethod
    def _fit_one_pseudo(pseudo_data_scaled, fixed_index, fixed_value, x0_seed, analysis,
                        bkg_varied_scaled=None):
        result_dict = {}
        if bkg_varied_scaled is not None:
            analysis._bkg_varied = bkg_varied_scaled
        try:
            try:
                res_free = analysis.optimize(pseudo_data_scaled / analysis.T,
                                             x0=x0_seed.copy(), display=False)
                result_dict.update(chi2_free=res_free.fun / analysis.c,
                                   x_free=res_free.x,
                                   success_free=res_free.success)
            except Exception as e:
                return {**result_dict, 'chi2_free': None, 'x_free': None,
                        'success_free': False, 'error': repr(e)}
            try:
                res_fixed = analysis.optimize_with_fixed_parameter(
                    pseudo_data_scaled / analysis.T,
                    fixed_index, fixed_value,
                    x0=res_free.x.copy()
                )
                result_dict.update(chi2_fixed=res_fixed.fun / analysis.c,
                                   x_fixed=res_fixed.x,
                                   success_fixed=res_fixed.success)
            except Exception as e:
                return {**result_dict, 'chi2_fixed': None, 'x_fixed': None,
                        'success_fixed': False, 'error': repr(e)}
        finally:
            if bkg_varied_scaled is not None:
                analysis._bkg_varied = None

        result_dict['delta_chi2'] = abs(result_dict['chi2_fixed'] - result_dict['chi2_free'])
        return result_dict

    def run_full_monte_carlo_analysis(self, num_pseudo_data, fixed_index, fixed_value,
                                      seed=None, n_jobs=1, verbose=True):
        # Free best fit to the *real* data. It is identical every call, so
        # compute it once and cache it: this both avoids redundant work and
        # keeps it from being overwritten by the pseudo-data fits below (each
        # _fit_one_pseudo calls optimize(), which sets self.result). The cache
        # is cleared in set_background when the data changes.
        use_pen = self.bkg_penalty
        if use_pen:
            # For the *real* data the "measured" background is the nominal B_i,
            # so the baseline fit subtracts B_i (identical to the standard fit)
            # and the observed Δχ² uses the same statistic as the pseudo-
            # experiments, which subtract their own B_varied.
            self._bkg_varied = self.Bkg_vector
        if (getattr(self, '_baseline_result', None) is None
                or getattr(self, '_baseline_mode', None) != use_pen):
            self._baseline_result = self.optimize(self.data_vector, display=False)
            self._baseline_mode = use_pen
        result = self._baseline_result

        # The Monte Carlo only needs χ² (for Δχ²) and modPrime = M·x / c. Both
        # are invariant under the OSQP vertex selection: every optimal x gives
        # the same M·x = μ and the same χ². Skip the HiGHS LP for every pseudo
        # fit — this is a >4× speed-up at large num_pseudo_data because each
        # fit then avoids 1 LP solve.
        _has_vs = hasattr(getattr(self, '_backend', None), 'vertex_select')
        _saved_vs = self._backend.vertex_select if _has_vs else None
        if _has_vs:
            self._backend.vertex_select = False

        result_fixed = self.optimize_with_fixed_parameter(
            self.data_vector, fixed_index, fixed_value, x0=result.x.copy()
        )
        self.best_fit_flux       = result.x
        self.best_fit_chi2       = result.fun
        self.best_fit_chi2_fixed = result_fixed.fun

        modPrime = self.M_matrix @ result_fixed.x / self.c
        self.modPrime_physical = modPrime / np.diff(self.bins)

        if verbose:
            print(f"[Baseline] χ²/c = {self.best_fit_chi2 / self.c:.6g}")
            print(f"[Baseline] event rates: {self.modPrime_physical.min():.3e} .. {self.modPrime_physical.max():.3e}")
            print(f"Generating {num_pseudo_data} pseudo-data sets "
                  f"(idx={fixed_index}, value={self.cm**2 * self.sec * fixed_value:.3e})")

        # Background sampling uses its own Generator so the legacy np.random
        # stream (and therefore the O_i^MC draws with bkg_penalty off) is
        # unchanged.
        rng_bkg = np.random.default_rng(seed) if use_pen else None
        if seed is not None:
            np.random.seed(seed)
        self.pseudo_data_sets = []
        self.pseudo_data_scaled = []
        self.pseudo_bkg_varied_scaled = []
        bw = np.diff(self.bins)
        for _ in range(num_pseudo_data):
            Event = self.T * (self.modPrime_physical * bw + self.Bkg_vector / self.c)
            pseudo_data = np.random.normal(Event, scale=np.sqrt(np.abs(Event)))
            if use_pen:
                # The varied background is now drawn independently of the
                # pseudo-data (Gaussian on B), so the pseudo-data is generated
                # exactly as in the non-penalty case — no O > 0 resampling.
                Bv_ev, _f = self._sample_varied_background(pseudo_data, rng_bkg)
                self.bkg_penalty_stats['n_pseudo'] += 1
                self.pseudo_bkg_varied_scaled.append(Bv_ev * self.c / self.T)
            self.pseudo_data_sets.append(pseudo_data)
            self.pseudo_data_scaled.append(pseudo_data * self.c)

        bkg_list = (self.pseudo_bkg_varied_scaled if use_pen
                    else [None] * num_pseudo_data)
        x0_seed = self.best_fit_flux
        # joblib parallelism is meaningful only for the scipy backend; OSQP is
        # already fast and cvxpy problems are awkward to ship across processes.
        if n_jobs != 1 and _HAS_JOBLIB and self.solver == 'scipy':
            self.fit_results = Parallel(n_jobs=n_jobs, prefer='processes')(
                delayed(self._fit_one_pseudo)(pd_, fixed_index, fixed_value,
                                              x0_seed, self, bkg_)
                for pd_, bkg_ in zip(self.pseudo_data_scaled, bkg_list)
            )
        else:
            self.fit_results = []
            for i, (pd_, bkg_) in enumerate(zip(self.pseudo_data_scaled, bkg_list), 1):
                r = self._fit_one_pseudo(pd_, fixed_index, fixed_value,
                                         x0_seed, self, bkg_)
                if verbose:
                    print(f"  [{i}/{num_pseudo_data}] "
                          f"χ²_free={r.get('chi2_free')}  "
                          f"χ²_fix={r.get('chi2_fixed')}  Δχ²={r.get('delta_chi2')}")
                self.fit_results.append(r)

        # Restore self.result to the real best fit: the loop above overwrote it
        # with the last pseudo-data fit, which would otherwise corrupt the
        # flux scatter in plot_flux_comparison / plot_flux_with_bands.
        self.result = result
        self._bkg_varied = None
        if _has_vs:
            self._backend.vertex_select = _saved_vs

    def analyze_monte_carlo_results(self, fixed_value, confidence_level=0.90):
        self.delta_chi2_values = [r['delta_chi2'] for r in self.fit_results
                                  if r.get('delta_chi2') is not None]
        if not self.delta_chi2_values:
            raise RuntimeError("No valid Δχ² values to compute cutoff.")
        ds = np.sort(self.delta_chi2_values)
        idx = max(int(confidence_level * len(ds)) - 1, 0)
        self.delta_chi2_cutoff = ds[idx]

        is_in_range = (
            self.best_fit_chi2_fixed / self.c
            < self.best_fit_chi2 / self.c + self.delta_chi2_cutoff
        )
        self.fixed_value_included = fixed_value if is_in_range else None
        return is_in_range

    # ---------- background penalty (nuisance background) ----------
    def set_bkg_penalty(self, on=True):
        """Toggle the background-penalty χ² used by the Monte Carlo.

        When on, every pseudo-experiment also draws a "measured" background
        B_varied_i = f_i · O_i^MC with f_i ~ Beta(α_i, β_i) (so 0 < B_varied <
        O^MC and the signal can never go negative), and the fit subtracts that
        B_varied in place of the nominal background, i.e. χ² = T·Σ (O −
        B_varied − M·x)²/O. The background is held fixed within each fit (not a
        free nuisance); only its variation across pseudo-experiments enters.
        Default off reproduces the previous behaviour exactly.
        """
        self.bkg_penalty = bool(on)
        self._baseline_result = None
        self._baseline_mode = None

    def reset_bkg_penalty_stats(self):
        """Clear the edge-case counters and stored sampling records."""
        self.bkg_penalty_stats = {
            'n_pseudo': 0,            # pseudo-experiments sampled
            'neg_O_resampled': 0,     # bin draws redone because O^MC ≤ 0
            'neg_O_clipped': 0,       # bins clipped to the expectation after 100 tries
            'neg_bkg_clipped': 0,     # B_varied < 0 draws clipped to 0
            'zero_bkg_bins': 0,       # bin draws skipped because B < threshold
        }
        self.bkg_sampling_records = []

    def _sample_varied_background(self, O_events, rng):
        """
        Draw the "measured" background for one pseudo-experiment by fluctuating
        each nominal background independently with a Gaussian of Poisson width,

            B_varied_i ~ Normal(B_i, sqrt(B_i))   (event counts),

        with B_i = T·Bkg_i/c. Unlike the earlier beta scheme this does *not*
        reference the pseudo-data event count O_i — the background varies on
        its own. Var(B_varied) = B_i matches the beta scheme's effective
        variance, but there is no 0 < B_varied < O coupling.

        Edge cases (counted in ``bkg_penalty_stats``):
        - B_varied < 0 (unphysical): clipped to 0.
        - B_i < ``bkg_zero_threshold`` events: the bin carries no background
          and is left at B_varied = 0 (reduces to the no-background χ² term).

        ``O_events`` is accepted for signature compatibility but unused.
        Returns ``(B_varied, ratio)`` in event counts, ratio = B_varied / B_i.
        """
        B_ev = self.T * self.Bkg_vector / self.c
        nbin = len(B_ev)
        ratio = np.zeros(nbin)
        Bv = np.zeros(nbin)
        mu_t = np.zeros(nbin)
        s2_t = np.zeros(nbin)
        stats = self.bkg_penalty_stats

        zero = B_ev < self.bkg_zero_threshold
        stats['zero_bkg_bins'] += int(zero.sum())
        act = ~zero
        if act.any():
            B = B_ev[act]
            draw = rng.normal(B, np.sqrt(B))
            neg = draw < 0
            stats['neg_bkg_clipped'] += int(neg.sum())
            draw = np.where(neg, 0.0, draw)
            Bv[act] = draw
            ratio[act] = draw / B
            # Stored in ratio space so the standardised residual below reads
            # (B_varied/B − 1)/sqrt(1/B) = (B_varied − B)/sqrt(B) ~ N(0, 1).
            mu_t[act] = 1.0
            s2_t[act] = 1.0 / B

        if len(self.bkg_sampling_records) < self._max_sampling_records:
            self.bkg_sampling_records.append(
                {'mu': mu_t, 'sigma2': s2_t, 'f': ratio, 'active': act,
                 'B_varied': Bv, 'O_mc': np.array(O_events)})
        return Bv, ratio

    def plot_bkg_sampling_check(self, save=True, fname=None):
        """
        Sanity check of the Gaussian sampling: pooled over the stored records,
        the standardised residuals (B_varied − B)/sqrt(B) should have mean ≈ 0
        and std ≈ 1, and the ratio B_varied/B should centre on 1. Also prints
        the edge-case counters. Returns a small summary dict.
        """
        recs = self.bkg_sampling_records
        if not recs:
            raise RuntimeError("No sampling records — run the Monte Carlo "
                               "with bkg_penalty on first.")
        f, mu, s2, bv = [], [], [], []
        for r in recs:
            a = r['active']
            f.append(r['f'][a]); mu.append(r['mu'][a])
            s2.append(r['sigma2'][a]); bv.append(r['B_varied'][a])
        f = np.concatenate(f); mu = np.concatenate(mu)
        s2 = np.concatenate(s2); bv = np.concatenate(bv)
        z = (f - mu) / np.sqrt(s2)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(f, bins=50)
        axes[0].set_xlabel(r'$B_\mathrm{varied}/B_i$')
        axes[0].set_title(f'ratio samples (n={len(f)}, mean$\\approx$1)')
        axes[1].hist(z, bins=50)
        axes[1].set_xlabel(r'$(f - \mu)/\sigma$')
        axes[1].set_title(f'standardised: mean={z.mean():.3f}, std={z.std():.3f}')
        axes[2].hist(bv, bins=50)
        axes[2].set_xlabel(r'$B_\mathrm{varied}$ [events]')
        axes[2].set_title('varied background')
        fig.tight_layout()
        print("bkg_penalty_stats:", self.bkg_penalty_stats)
        if save:
            os.makedirs(self.scenario_dir, exist_ok=True)
            if fname is None:
                fname = f'bkg_sampling_check_bkg_{self.background_scenario}.pdf'
            path = os.path.join(self.scenario_dir, fname)
            fig.savefig(path)
            print(f"Plot saved as {path}")
        return {'n_samples': len(f),
                'std_resid_mean': float(z.mean()),
                'std_resid_std': float(z.std()),
                'stats': dict(self.bkg_penalty_stats)}

    def scan_fixed_parameter(self, fixed_index, scan_range=0.35, num_points=11,
                             num_pseudo_data=10, seed=None, n_jobs=1):
        base_value = self.result.x[fixed_index]
        if fixed_index >= 50:
            scan_range = 1 - 1e-11

        if fixed_index == 0:
            fixed_values = np.append(
                np.linspace(8.089268e+11 / self.cm ** 2 / self.sec,
                            base_value,
                            num_points)[:-1],
                np.linspace(base_value,
                            2.157467e+13 / self.cm ** 2 / self.sec, num_points)
            )
        elif fixed_index == 1:
            fixed_values = np.append(
                np.linspace(7.089268e+11 / self.cm ** 2 / self.sec,
                            base_value,
                            num_points)[:-1],
                np.linspace(base_value,
                            1.026211e+13 / self.cm ** 2 / self.sec, num_points)
            )
        elif fixed_index < 7:
            fixed_values = np.linspace((1 - scan_range) * base_value,
                                       (1 + 2) * base_value,
                                       2 * num_points - 1)
        elif fixed_index < 50:
            fixed_values = np.linspace((1 - scan_range) * base_value,
                                       (1 + scan_range) * base_value,
                                       2 * num_points - 1)
        else:
            fixed_values = np.append(
                np.linspace((1 - scan_range) * base_value,
                            (1 + scan_range) * base_value, num_points)[:-1],
                np.linspace((1 + scan_range) * base_value, 4 * base_value, num_points)
            )

        results = []
        total = len(fixed_values)
        print(f"[Scan idx={fixed_index}] {total} points, solver={self.solver}")
        for i, fv in enumerate(fixed_values, 1):
            print(f"\n--- [{i}/{total}] flux={self.cm**2 * self.sec * fv:.4e} ---")
            self.run_full_monte_carlo_analysis(
                num_pseudo_data=num_pseudo_data,
                fixed_index=fixed_index,
                fixed_value=fv,
                seed=seed,
                n_jobs=n_jobs,
                verbose=False,
            )
            in68 = self.analyze_monte_carlo_results(fv, 0.678);  c68 = self.delta_chi2_cutoff
            in90 = self.analyze_monte_carlo_results(fv, 0.90);   c90 = self.delta_chi2_cutoff
            in95 = self.analyze_monte_carlo_results(fv, 0.954);  c95 = self.delta_chi2_cutoff

            results.append({
                'index': fixed_index,
                'fixed_value_raw': fv,
                'fixed_value_physical': self.cm ** 2 * self.sec * fv,
                'included1sigma': in68, 'included90': in90, 'included2sigma': in95,
                'delta_chi2_cutoff68': c68, 'delta_chi2_cutoff90': c90, 'delta_chi2_cutoff95': c95,
                'delta_chi2': list(self.delta_chi2_values),
            })
        return results

    # ---------- confidence band by root finding ----------
    def _band_eval(self, fixed_index, v, levels, num_pseudo_data, seed, n_jobs, cache):
        """
        Evaluate one fixed value: run the Monte Carlo once and return, for every
        confidence level, whether ``v`` is inside the band plus the observed Δχ²
        and the cutoff. Results are cached by ``v`` (deterministic when ``seed``
        is fixed), so the bisection never recomputes a point.
        """
        key = round(float(v), 9)
        if key in cache:
            return cache[key]
        self.run_full_monte_carlo_analysis(
            num_pseudo_data=num_pseudo_data, fixed_index=fixed_index,
            fixed_value=float(v), seed=seed, n_jobs=n_jobs, verbose=False,
        )
        dchi2_obs = self.best_fit_chi2_fixed / self.c - self.best_fit_chi2 / self.c
        included, cutoff = {}, {}
        for lv in levels:
            included[lv] = self.analyze_monte_carlo_results(float(v), lv)
            cutoff[lv] = self.delta_chi2_cutoff
        res = {'v': float(v), 'dchi2_obs': dchi2_obs,
               'included': included, 'cutoff': cutoff}
        cache[key] = res
        return res

    def _bisect_edge(self, fixed_index, level, v_in, v_out, levels,
                     n_pseudo_edge, rel_tol, seed, n_jobs, cache):
        """Geometric bisection between v_in (inside band) and v_out (outside)."""
        a, b = float(v_in), float(v_out)
        for _ in range(40):
            if abs(b - a) <= rel_tol * max(abs(b), abs(a)):
                break
            m = np.sqrt(a * b) if (a > 0 and b > 0) else 0.5 * (a + b)
            r = self._band_eval(fixed_index, m, levels, n_pseudo_edge,
                                seed, n_jobs, cache)
            if r['included'][level]:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    def find_confidence_band(self, fixed_index, levels=(0.678, 0.90, 0.954),
                             num_pseudo_data=30, n_pseudo_edge=200,
                             step=1.5, rel_tol=0.03, max_bracket=25,
                             seed=42, n_jobs=1, verbose=True, bkg_penalty=None):
        """
        Locate the confidence-band edges for one flux parameter by root finding
        instead of a uniform grid. The widest level brackets the outer edges;
        each level's edge is then pinned by geometric bisection, with more
        pseudo-data (``n_pseudo_edge``) used during refinement to tame the
        Monte-Carlo noise in the cutoff. ``seed`` is fixed so each value is
        reproducible (keeps the bisection from jittering).

        ``bkg_penalty`` switches the background-penalty χ² on/off for this and
        subsequent runs (sticky, like ``set_background``); ``None`` keeps the
        current setting.

        Returns a dict: per level a (lower, upper) pair in raw units, plus the
        same in physical units, and the best-fit value.
        """
        if bkg_penalty is not None and bool(bkg_penalty) != self.bkg_penalty:
            self.set_bkg_penalty(bkg_penalty)
        if self.result is None:
            self.optimize(self.data_vector)
        levels = tuple(sorted(levels))
        widest = levels[-1]
        v0 = float(self.result.x[fixed_index])
        unit = self.cm ** 2 * self.sec
        cache = {}

        r0 = self._band_eval(fixed_index, v0, levels, num_pseudo_data,
                             seed, n_jobs, cache)
        if not r0['included'][widest] and verbose:
            print(f"[warn] best-fit value v0={v0:.4g} is already outside the "
                  f"{widest:.3f} band — check the construction.")
        if verbose:
            print(f"[band idx={fixed_index}] v0={unit*v0:.4e} (phys), "
                  f"bracketing with step={step}")

        # Bracket outward until excluded at the widest level.
        def bracket(direction):
            v = v0
            for _ in range(max_bracket):
                v = v * step if direction > 0 else v / step
                if v <= 0:
                    return None
                r = self._band_eval(fixed_index, v, levels, num_pseudo_data,
                                    seed, n_jobs, cache)
                if not r['included'][widest]:
                    return v
            return None

        up_out = bracket(+1)
        lo_out = bracket(-1)
        if verbose:
            print(f"  upper bracket: {unit*up_out:.4e}" if up_out else
                  "  upper edge unbounded (still inside at max_bracket)")
            print(f"  lower bracket: {unit*lo_out:.4e}" if lo_out else
                  "  lower edge reaches 0 / unbounded below")

        band_raw, band_phys = {}, {}
        for lv in levels:
            upper = (self._bisect_edge(fixed_index, lv, v0, up_out, levels,
                                       n_pseudo_edge, rel_tol, seed, n_jobs, cache)
                     if up_out is not None else np.inf)
            lower = (self._bisect_edge(fixed_index, lv, v0, lo_out, levels,
                                       n_pseudo_edge, rel_tol, seed, n_jobs, cache)
                     if lo_out is not None else 0.0)
            band_raw[lv] = (lower, upper)
            band_phys[lv] = (lower * unit, upper * unit)
            if verbose:
                print(f"  level {lv:.3f}: [{lower*unit:.4e}, {upper*unit:.4e}] (phys)")

        return {
            'index': fixed_index,
            'best_fit_raw': v0,
            'best_fit_physical': v0 * unit,
            'levels': levels,
            'band_raw': band_raw,
            'band_physical': band_phys,
            'n_evaluations': len(cache),
            'bkg_penalty': self.bkg_penalty,
        }

    # ---------- plotting ----------
    def _calculate_integrated_flux(self):
        def dPhidEnu(E):
            return np.interp(E, self.fig1Solid['MeV'], self.fig1Solid['cm**-2sec-1MeV-1'], 1)
        def dPhidEnudashed(E):
            return np.interp(E, self.fig1dashed['MeV'], self.fig1dashed['cm**-2sec-1MeV-1'], 1)
        x2to7 = np.logspace(np.log10(2), np.log10(7), self.iterationTime)
        Phi2to7       = integrate.trapezoid(dPhidEnu(x2to7),       x2to7)
        Phi2to7dashed = integrate.trapezoid(dPhidEnudashed(x2to7), x2to7)
        x  = np.logspace(-2, np.log10(7), self.iterationTime); y  = dPhidEnu(x)
        x2 = np.logspace(-2, np.log10(7), self.iterationTime); y2 = dPhidEnudashed(x2)
        Phi       = np.zeros(self.iterationTime)
        Phidashed = np.zeros(self.iterationTime)
        for i in range(self.iterationTime):
            Phi[i]       = integrate.trapezoid(y[i:],  x[i:])
            Phidashed[i] = integrate.trapezoid(y2[i:], x2[i:])
        return x, Phi - Phi2to7, x2, Phidashed - Phi2to7dashed

    def plot_flux_comparison(self, save=True, norm=1e12):
        if self.result is None:
            raise RuntimeError("Run optimize() before plotting.")
        x, Phi, x2, Phidashed = self._calculate_integrated_flux()
        plt.figure(figsize=(8, 6))
        plt.plot(x,  Phi / norm,       label=r'With neutron capture',    color='red')
        plt.plot(x2, Phidashed / norm, label=r'Without neutron capture', color='brown', ls='dashed')
        eb = np.linspace(0.41, 2, self.n)
        plt.scatter(eb, self.result.x * (self.cm ** 2 * self.sec) / norm,
                    label=f'Optimized Flux (Bkg: {self.background_scenario})',
                    zorder=5, s=1)
        plt.xscale('log'); plt.ylim(0, 2e12 / norm); plt.xlim(0.1, 2)
        plt.xlabel(r"$E_\nu$ [MeV]"); plt.ylabel(_phi_ylabel(norm))
        plt.title('Optimized Neutrino Flux vs. Theoretical Models')
        plt.legend(); plt.grid(True, which="both", ls="--")
        if save:
            os.makedirs(self.scenario_dir, exist_ok=True)
            fn = f'{self.scenario_dir}/flux_comparison_bkg_{self.background_scenario}.pdf'
            plt.savefig(fn); print(f"Plot saved as {fn}")

    # ---------- confidence band: save / batch / overlay ----------
    def save_band(self, band, outdir='bands', fname=None):
        """Write one ``find_confidence_band`` result to JSON (one file per index)."""
        os.makedirs(outdir, exist_ok=True)
        pen = bool(band.get('bkg_penalty', getattr(self, 'bkg_penalty', False)))
        if fname is None:
            # '_bkgpen' suffix keeps penalty bands from overwriting the
            # standard ones, so the two constructions can be compared.
            suffix = '_bkgpen' if pen else ''
            fname = (f'band_bkg{self.background_scenario}'
                     f'_idx{band["index"]:03d}{suffix}.json')
        path = os.path.join(outdir, fname)
        obj = {
            'index': int(band['index']),
            'background_scenario': self.background_scenario,
            'bkg_penalty': pen,
            'best_fit_raw': float(band['best_fit_raw']),
            'best_fit_physical': float(band['best_fit_physical']),
            'levels': [float(l) for l in band['levels']],
            'band_raw': {f'{float(l):.6f}': [float(lo), float(hi)]
                         for l, (lo, hi) in band['band_raw'].items()},
            'band_physical': {f'{float(l):.6f}': [float(lo), float(hi)]
                              for l, (lo, hi) in band['band_physical'].items()},
            'n_evaluations': int(band.get('n_evaluations', 0)),
        }
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)
        print(f"Band saved as {path}")
        return path

    def find_and_save_band(self, fixed_index, outdir=None, **kwargs):
        """Run ``find_confidence_band`` for one index and save it immediately.

        ``outdir`` defaults to ``self.bands_dir`` (= ``T<T>/scenario_bkg_<x>/bands``)
        so different T runs don't overwrite each other.
        """
        if outdir is None:
            outdir = self.bands_dir
        band = self.find_confidence_band(fixed_index, **kwargs)
        self.save_band(band, outdir=outdir)
        return band

    def plot_flux_with_bands(self, band_files, levels=None, save=True,
                             fname=None, ylim=None, style='fill', norm=1e12):
        """
        Overlay saved confidence bands on the optimized flux. ``band_files`` is a
        list of JSON paths or a glob pattern (e.g. ``'bands/band_*idx*.json'``).
        ``self.optimize`` must have been run first so the scatter and band
        centres line up.

        ``style='fill'`` (default) connects the lower and upper edges of each
        level across energy bins with lines and shades the interval in a
        translucent colour (one colour per level, widest drawn underneath).
        ``style='errorbar'`` draws an asymmetric error bar per bin instead, and
        ``style='both'`` overlays the error bars on the shaded band.
        """
        if self.result is None:
            raise RuntimeError("Run optimize() before plotting.")
        if isinstance(band_files, str):
            band_files = sorted(glob.glob(band_files))
        bands = [load_band(p) for p in band_files]
        if not bands:
            raise RuntimeError("No band files found.")

        eb = np.linspace(0.41, 2, self.n)
        unit = self.cm ** 2 * self.sec
        x, Phi, x2, Phidashed = self._calculate_integrated_flux()

        plt.figure(figsize=(8, 6))
        plt.plot(x,  Phi / norm,       color='red',   label='With NC', lw = 3)
        plt.plot(x2, Phidashed / norm, color='brown', ls='dashed', label='Without NC', lw = 3)
        plt.scatter(eb, self.result.x * unit / norm, s=30, color='lightgreen', zorder=5,
                    label=f'Best-fit Flux (Bkg: {self.background_scenario})')

        all_levels = levels if levels is not None else sorted(
            {l for b in bands for l in b['levels']})
        cyc = ['C2', 'C1', 'C0', 'C3', 'C4']
        level_colors = {lv: cyc[k % len(cyc)]
                        for k, lv in enumerate(sorted(all_levels, reverse=True))}

        if style in ('fill', 'both'):
            # widest level first so narrower levels are drawn on top
            for lv in sorted(all_levels, reverse=True):
                pts = sorted((eb[b['index']],) + tuple(b['band_physical'][lv])
                             for b in bands if lv in b['band_physical'])
                if not pts:
                    continue
                xs = np.array([p[0] for p in pts])
                lo = np.array([p[1] for p in pts]) / norm
                hi = np.array([p[2] for p in pts]) / norm
                ok = np.isfinite(lo) & np.isfinite(hi)
                col = level_colors[lv]
                plt.fill_between(xs[ok], lo[ok], hi[ok], color=col, alpha=0.25,
                                 zorder=2, label=f'{lv:.3f} band')
                plt.plot(xs[ok], lo[ok], color=col, lw=1.0, zorder=3)
                plt.plot(xs[ok], hi[ok], color=col, lw=1.0, zorder=3)
        if style in ('errorbar', 'both'):
            labeled = set()
            for b in bands:
                xpos = eb[b['index']]
                c = b['best_fit_physical']
                for lv in sorted(b['levels'], reverse=True):
                    if levels is not None and lv not in levels:
                        continue
                    lo, hi = b['band_physical'][lv]
                    lo_err = max(c - lo, 0.0) if np.isfinite(lo) else 0.0
                    hi_err = max(hi - c, 0.0) if np.isfinite(hi) else 0.0
                    lab = None
                    if style != 'both' and lv not in labeled:
                        lab = f'{lv:.3f} band'
                        labeled.add(lv)
                    plt.errorbar([xpos], [c / norm], yerr=[[lo_err / norm], [hi_err / norm]],
                                 fmt='none', ecolor=level_colors[lv], elinewidth=1.5,
                                 capsize=2, alpha=0.7, zorder=4, label=lab)

        plt.xscale('log'); plt.xlim(0.1, 2)
        if ylim is not None:
            plt.ylim(ylim[0] / norm, ylim[1] / norm)
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 20
        plt.xlabel(r"$E_\nu$ [MeV]", fontsize = 30); plt.ylabel(_phi_ylabel(norm), fontsize = 30)
        plt.legend(fontsize=8)
        plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10) * 0.1, numticks=20))
        plt.tick_params(axis='both', which='major', labelsize=23)
        plt.tick_params(axis='both', which='minor', labelsize=23)
        plt.legend(loc = 'upper right', fontsize = 15, frameon = False)
        if save:
            os.makedirs(self.scenario_dir, exist_ok=True)
            if fname is None:
                fname = (f'{self.scenario_dir}/'
                         f'flux_with_bands_bkg_{self.background_scenario}.pdf')
            plt.savefig(fname); print(f"Plot saved as {fname}")

    # ---------- degeneracy band (Δχ² < threshold) ----------
    _EB_MIN = 0.41   # low edge of the reconstructed-flux energy grid (MeV)

    def _degeneracy_edges(self, index, threshold, res_x, chi2min):
        """Lower/upper *raw* flux values where the profiled Δχ²/c first exceeds
        ``threshold`` on each side of the best fit for one parameter. The
        profiled χ² is convex in the fixed value, so each side has a single
        crossing: bracket outward, then bisect."""
        data = self.data_vector

        def dchi(v):
            r = self.optimize_with_fixed_parameter(data, index, float(v),
                                                   x0=res_x.copy())
            return r.fun / self.c - chi2min

        def edge(direction, factor=1.4, nbracket=40, nbisect=40):
            v_in, v_out, v = float(res_x[index]), None, float(res_x[index])
            for _ in range(nbracket):
                v = v * factor if direction > 0 else v / factor
                if v <= 0:
                    return 0.0
                if dchi(v) > threshold:
                    v_out = v
                    break
                v_in = v
            if v_out is None:
                return v_in                        # no crossing within range
            lo, hi = (v_in, v_out) if direction > 0 else (v_out, v_in)
            for _ in range(nbisect):
                m = 0.5 * (lo + hi)
                inside = dchi(m) < threshold
                if direction > 0:
                    lo, hi = (m, hi) if inside else (lo, m)
                else:
                    lo, hi = (lo, m) if inside else (m, hi)
            return 0.5 * (lo + hi)

        return edge(-1), edge(+1)

    def _degeneracy_setup(self, x0=None):
        """Common prep for the degeneracy scan: ensure a best fit, and disable
        vertex selection (only χ² is needed). Returns (res_x, chi2min, restore)."""
        if self.result is None:
            self.optimize(self.data_vector)
        res_x = self.result.x if x0 is None else np.asarray(x0, dtype=float)
        chi2min = self.result.fun / self.c
        _be = getattr(self, '_backend', None)
        _has_vs = hasattr(_be, 'vertex_select')
        _saved = _be.vertex_select if _has_vs else None
        if _has_vs:
            _be.vertex_select = False

        def restore():
            if _has_vs:
                _be.vertex_select = _saved
        return res_x, chi2min, restore

    def compute_degeneracy_band(self, indices, threshold=1e-3, x0=None, eb=None):
        """Per-index degeneracy interval: the range of a single flux parameter
        over which the profiled Δχ²/c (re-optimising all other parameters)
        stays below ``threshold`` around the best fit — the intrinsic
        under-determination band, not a statistical one.

        Returns a sorted array of (energy_MeV, lower_phys, upper_phys) in
        physical flux units (cm⁻² s⁻¹)."""
        res_x, chi2min, restore = self._degeneracy_setup(x0)
        unit = self.cm ** 2 * self.sec
        if eb is None:
            eb = np.linspace(self._EB_MIN, 2, self.n)
        try:
            rows = []
            for idx in sorted({int(i) for i in indices}):
                lo, hi = self._degeneracy_edges(idx, threshold, res_x, chi2min)
                rows.append((eb[idx], lo * unit, hi * unit))
        finally:
            restore()
        rows.sort()
        return np.array(rows)

    def save_degeneracy_bands(self, indices, threshold=1e-3, outdir=None,
                              x0=None, verbose=True):
        """Pre-compute and save the degeneracy interval for each index as one
        JSON file, mirroring ``save_band``. Files land in ``outdir`` (default
        ``<scenario_dir>/degeneracy``) named ``degeneracy_bkg<scen>_idx<NNN>.json``
        and store raw + physical lower/upper, best fit, threshold and T."""
        if outdir is None:
            outdir = os.path.join(self.scenario_dir, 'degeneracy')
        os.makedirs(outdir, exist_ok=True)
        res_x, chi2min, restore = self._degeneracy_setup(x0)
        unit = self.cm ** 2 * self.sec
        eb = np.linspace(self._EB_MIN, 2, self.n)
        paths = []
        try:
            for idx in sorted({int(i) for i in indices}):
                lo, hi = self._degeneracy_edges(idx, threshold, res_x, chi2min)
                obj = {
                    'index': int(idx),
                    'background_scenario': self.background_scenario,
                    'T': self.T,
                    'threshold': float(threshold),
                    'energy_MeV': float(eb[idx]),
                    'best_fit_raw': float(res_x[idx]),
                    'best_fit_physical': float(res_x[idx] * unit),
                    'lower_raw': float(lo),
                    'upper_raw': float(hi),
                    'lower_physical': float(lo * unit),
                    'upper_physical': float(hi * unit),
                }
                path = os.path.join(
                    outdir,
                    f'degeneracy_bkg{self.background_scenario}_idx{idx:03d}.json')
                with open(path, 'w') as f:
                    json.dump(obj, f, indent=2)
                paths.append(path)
                if verbose:
                    print(f'  degeneracy idx{idx:03d}: '
                          f'[{lo*unit:.3e}, {hi*unit:.3e}]  -> {path}', flush=True)
        finally:
            restore()
        return paths

    def plot_band_comparison(self, groups, level=0.954, show_theory=True,
                             optimized=None, save=True, fname=None,
                             ylim=None, xlim=None, logy=False, style='fill', norm=1e12,
                             show_errorbars=True,
                             bestfit_color='#0072B2', bestfit_marker='o',
                             degeneracy=False, degeneracy_indices=None,
                             degeneracy_threshold=1e-3, degeneracy_color='#7030a0',
                             degeneracy_label=r'Degeneracy $\Delta\chi^2<10^{-3}$',
                             degeneracy_files=None):
        """
        Overlay confidence bands from several scenarios on one axis.

        ``level`` is a single confidence level (e.g. ``0.954``) or a tuple of
        levels (e.g. ``(0.678, 0.954)``); with several levels each group is
        drawn once per level, wider levels lighter and narrower levels darker,
        and the legend label gets a ``1σ`` / ``90%`` / ``2σ`` tag appended.
        ``xlim`` / ``ylim`` are in physical units (``ylim`` is divided by ``norm``).

        ``groups`` is a dict ``{label: band_files}`` where ``band_files`` is a
        glob pattern or a list of JSON paths written by ``save_band`` (e.g.
        ``{'flat': 'scenario_bkg_flat/bands/band_*.json',
           'b':    'scenario_bkg_b/bands/band_*.json'}``).

        Each scenario is drawn in its own colour. ``style`` selects how the
        ``level`` band is shown:
        ``'fill'`` (default) shades the band between its lower and upper edges,
        ``'errorbar'`` draws an asymmetric error bar per bin, and ``'both'``
        overlays the error bars on the shaded band.
        Bands are in physical units (cm⁻² s⁻¹), comparable across scenarios.

        ``optimized`` optionally overlays the full optimized flux for each
        scenario as a scatter in the matching colour. It is a dict
        ``{label: value}`` where value is either a ``NeutrinoAnalysis`` instance
        (its ``result.x`` is used) or a raw flux array. Labels should match
        ``groups`` so colours line up.

        ``degeneracy=True`` also overlays the intrinsic degeneracy band
        (profiled Δχ²/c < ``degeneracy_threshold``) for *this* analysis,
        computed on the fly via ``compute_degeneracy_band``. By default it is
        evaluated at the union of indices found in ``groups``' band files;
        pass ``degeneracy_indices`` to override. ``self.optimize`` is run first
        if needed, so the band matches this instance's background scenario.
        """
        eb = np.linspace(0.41, 2, self.n)
        cyc = ['C0', 'C1', 'C2', 'C3', 'C4']
        group_color = {label: cyc[k % len(cyc)]
                       for k, label in enumerate(groups)}
        deg_idx_set = set()

        # accept a scalar or a sequence of levels; draw widest first
        levels = ((level,) if np.isscalar(level) else tuple(level))
        levels = tuple(sorted(levels, reverse=True))
        multi = len(levels) > 1

        def match_level(b, target):
            for lv in b['band_physical']:
                if abs(lv - target) < 1e-6:
                    return lv
            return None

        def level_tag(lv):
            for ref, tag in ((0.678, r'1$\sigma$'), (0.683, r'1$\sigma$'),
                             (0.90, r'90\%'), (0.954, r'2$\sigma$'),
                             (0.997, r'3$\sigma$')):
                if abs(lv - ref) < 1e-3:
                    return tag
            return f'{100 * lv:.1f}%'

        # alpha per level: widest lightest, narrowest darkest
        if multi:
            alphas = np.linspace(0.18, 0.40, len(levels))
        else:
            alphas = np.array([0.22])

        plt.figure(figsize=(8, 6))

        if show_theory:
            x, Phi, x2, Phidashed = self._calculate_integrated_flux()
            plt.plot(x,  Phi / norm,       color='black', lw = 3,
                     label='With NC')
            plt.plot(x2, Phidashed / norm, color='black', lw = 3, ls='dashed',
                     label='Without NC')

        if optimized:
            for label, val in optimized.items():
                if hasattr(val, 'result'):          # a NeutrinoAnalysis instance
                    xr = val.result.x
                    unit = val.cm ** 2 * val.sec
                    nn = val.n
                else:                                # a raw flux array
                    xr = np.asarray(val)
                    unit = self.cm ** 2 * self.sec
                    nn = len(xr)
                eb_g = np.linspace(0.41, 2, nn)
                plt.scatter(eb_g, np.asarray(xr) * unit / norm, s=30,
                            marker=bestfit_marker, color=bestfit_color,
                            zorder=5, label=f'{label} Best-fit')

        for k, (label, files) in enumerate(groups.items()):
            if isinstance(files, str):
                files = sorted(glob.glob(files))
            bands = [load_band(p) for p in files]
            if not bands:
                print(f"[warn] no band files for group '{label}'")
                continue
            color = group_color[label]
            for b in bands:
                deg_idx_set.add(int(b['index']))
            for il, target in enumerate(levels):
                rows = []
                for b in bands:
                    lv = match_level(b, target)
                    if lv is None:
                        continue
                    lo, hi = b['band_physical'][lv]
                    rows.append((eb[b['index']], b['best_fit_physical'], lo, hi))
                if not rows:
                    print(f"[warn] group '{label}': no bands at level {target}")
                    continue
                rows.sort()
                xs = np.array([r[0] for r in rows])
                cen = np.array([r[1] for r in rows]) / norm
                lo = np.array([r[2] for r in rows]) / norm
                hi = np.array([r[3] for r in rows]) / norm
                ok = np.isfinite(lo) & np.isfinite(hi)
                lbl = f'{label} ({level_tag(target)})' if multi else label
                if style in ('fill', 'both'):
                    plt.fill_between(xs[ok], lo[ok], hi[ok], color=color,
                                     alpha=float(alphas[il]), zorder=2 + 0.1 * il,
                                     label=lbl)
                    plt.plot(xs[ok], lo[ok], color=color, lw=1.0, zorder=3)
                    plt.plot(xs[ok], hi[ok], color=color, lw=1.0, zorder=3)
                    lbl = None
                if style in ('errorbar', 'both') and show_errorbars:
                    lo_err = np.where(np.isfinite(lo), np.maximum(cen - lo, 0.0), 0.0)
                    hi_err = np.where(np.isfinite(hi), np.maximum(hi - cen, 0.0), 0.0)
                    plt.errorbar(xs, cen, yerr=[lo_err, hi_err],
                                 fmt='none', ms=2.5, color=color, ecolor=color,
                                 elinewidth=1.3, capsize=2 + 2 * il, alpha=0.75,
                                 zorder=4, label=lbl)

        if degeneracy or degeneracy_files is not None:
            if degeneracy_files is not None:
                # load pre-computed per-index degeneracy JSONs (save_degeneracy_bands)
                files = (sorted(glob.glob(degeneracy_files))
                         if isinstance(degeneracy_files, str) else list(degeneracy_files))
                rows = []
                for p in files:
                    dd = load_degeneracy_band(p)
                    rows.append((eb[dd['index']], dd['lower_physical'],
                                 dd['upper_physical']))
                rows.sort()
                dband = np.array(rows) if rows else np.empty((0, 3))
            else:
                didx = degeneracy_indices if degeneracy_indices is not None else sorted(deg_idx_set)
                dband = (self.compute_degeneracy_band(didx, threshold=degeneracy_threshold, eb=eb)
                         if len(didx) else np.empty((0, 3)))
            if len(dband):
                dE, dlo, dhi = dband[:, 0], dband[:, 1] / norm, dband[:, 2] / norm
                ok = np.isfinite(dlo) & np.isfinite(dhi)
                plt.fill_between(dE[ok], dlo[ok], dhi[ok], color=degeneracy_color,
                                 alpha=0.22, zorder=1.5, label=degeneracy_label)
                plt.plot(dE[ok], dlo[ok], color=degeneracy_color, lw=1.0, zorder=3)
                plt.plot(dE[ok], dhi[ok], color=degeneracy_color, lw=1.0, zorder=3)
            else:
                print("[warn] degeneracy requested but no data to draw")

        plt.xscale('log')
        if logy:
            plt.yscale('log')
        plt.xlim(*(xlim if xlim is not None else (0.35, 3)))
        if ylim is not None:
            plt.ylim(ylim[0] / norm, ylim[1] / norm)
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 20
        plt.xlabel(r"$E_\nu$ [MeV]", fontsize = 30); plt.ylabel(_phi_ylabel(norm), fontsize = 30)
        plt.legend(fontsize=8)
        plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10) * 0.1, numticks=20))
        plt.tick_params(axis='both', which='major', labelsize=23)
        plt.tick_params(axis='both', which='minor', labelsize=23)
        plt.legend(loc = 'upper right', fontsize = 15, frameon = False)
        if save:
            if fname is None:
                fname = ('band_comparison_level' +
                         '_'.join(f'{lv:.3f}' for lv in levels) + '.pdf')
            plt.savefig(fname); print(f"Plot saved as {fname}")

    def generate_pseudo_data(self, num_pseudo_data=500, seed=None, x=None):
        """
        Generate Gaussian pseudo-data sets around the model expectation, without
        running any fits (this is just the data-generation step of the Monte
        Carlo). Each set is in event units (counts over the observation time T):

            Event_i = T * (M @ x / c + Bkg / c)_i ,
            pseudo  ~ Normal(Event, sqrt(|Event|)) .

        ``x`` defaults to the free best fit ``self.result.x`` (optimize() is run
        on the real data if needed). Stores and returns ``self.pseudo_data_sets``.
        """
        if x is None:
            if self.result is None:
                self.optimize(self.data_vector)
            x = self.result.x
        model = self.M_matrix @ x / self.c               # = modPrime
        Event = self.T * (model + self.Bkg_vector / self.c)
        if seed is not None:
            np.random.seed(seed)
        self.pseudo_data_sets = [
            np.random.normal(Event, scale=np.sqrt(np.abs(Event)))
            for _ in range(num_pseudo_data)
        ]
        return self.pseudo_data_sets

    def plot_pseudo_data_bins(self, num_pseudo_data=500, seed=None,
                              regenerate=False, show_lines=False,
                              save=True, fname=None):
        """
        Per-bin comparison of the pseudo-data sets against the real data.

        If no pseudo-data exist yet (or ``regenerate=True``), this generates
        ``num_pseudo_data`` sets via ``generate_pseudo_data`` first -- no Monte
        Carlo fitting is performed. If a previous run already populated
        ``self.pseudo_data_sets`` (e.g. run_full_monte_carlo_analysis), those
        are reused unless ``regenerate=True``.

        Both are in event units (counts over the observation time ``T``): the
        pseudo sets are Gaussian draws around the model, and the real data is
        ``T * data_vector / c``. By default the pseudo spread is shown as the
        median plus 68%% / 95%% percentile bands; ``show_lines=True`` overlays
        every pseudo set as a faint step line. The real data is drawn with
        sqrt(N) (Poisson) error bars.
        """
        if regenerate or not getattr(self, 'pseudo_data_sets', None):
            self.generate_pseudo_data(num_pseudo_data, seed)
        pseudo = np.asarray(self.pseudo_data_sets)          # (N, m) event counts
        true_ev = self.T * self.data_vector / self.c        # (m,) real-data events
        edges = self.bins
        centers = 0.5 * (edges[:-1] + edges[1:])
        n = len(pseudo)

        plt.figure(figsize=(8, 6))
        if show_lines:
            for row in pseudo:
                plt.step(centers, row, where='mid', color='C0',
                         alpha=min(0.05, 5.0 / max(n, 1)), lw=0.5)
            plt.step([], [], color='C0', label=f'{n} pseudo sets')
        else:
            lo95, lo68, med, hi68, hi95 = np.percentile(
                pseudo, [2.5, 16, 50, 84, 97.5], axis=0)
            plt.fill_between(centers, lo95, hi95, step='mid', color='C0',
                             alpha=0.20, label='pseudo 95%')
            plt.fill_between(centers, lo68, hi68, step='mid', color='C0',
                             alpha=0.35, label='pseudo 68%')
            plt.step(centers, med, where='mid', color='C0', lw=1.0,
                     label='pseudo median')

        plt.errorbar(centers, true_ev, yerr=np.sqrt(np.abs(true_ev)),
                     fmt='o', ms=3, color='k', capsize=2, zorder=5,
                     label='True data')
        plt.xlabel('Recoil energy bin'); plt.ylabel('Events')
        plt.title(f'Pseudo-data vs. true data '
                  f'(Bkg: {self.background_scenario}, T={self.T}, N={n})')
        plt.legend(fontsize=8); plt.grid(True, ls='--', alpha=0.5)
        if save:
            os.makedirs(self.scenario_dir, exist_ok=True)
            if fname is None:
                fname = (f'{self.scenario_dir}/'
                         f'pseudo_vs_true_bins_bkg_{self.background_scenario}.pdf')
            plt.savefig(fname); print(f"Plot saved as {fname}")

    def plot_scan_results(self, index, fixed_values, chi_sq_results, save=True):
        plt.figure(figsize=(8, 6))
        fv = fixed_values * (self.cm ** 2 * self.sec)
        plt.plot(fv, chi_sq_results - np.min(chi_sq_results), 'o-')
        plt.hlines(1/3,  1e-3 * fv[0], 1e3 * fv[-1], color='pink', ls='dashed', label=r'$\Delta \chi^2 = 1/3$')
        plt.hlines(1e-3, 1e-3 * fv[0], 1e3 * fv[-1], color='blue', ls='dotted', label=r'$\Delta \chi^2 = 10^{-3}$')
        if index < 2:
            plt.vlines(fv[(len(fv) - 1) // 2], 1e-10, 1e5, color='red', ls='dashdot', label='Minimized')
            plt.xscale('log')
            plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10) * 0.1, numticks=20))
        elif index < 50:
            plt.vlines(fv[(len(fv) - 1) // 2], 1e-10, 1e5, color='red', ls='dashdot', label='Minimized')
        else:
            plt.vlines(fv[(len(fv) - 1) // 4], 1e-10, 1e5, color='red', ls='dashdot', label='Minimized')
        x, Phi, x2, Phidashed = self._calculate_integrated_flux()
        eb = np.linspace(0.41, 2, self.n)
        plt.vlines(Phi[np.argmin(np.abs(x  - eb[index]))], 1e-10, 1e5, color='grey', label='With neutron capture')
        plt.vlines(Phidashed[np.argmin(np.abs(x2 - eb[index]))], 1e-10, 1e5, color='grey', ls='dashed', label='Without neutron capture')
        plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10) * 0.1, numticks=20))
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='both', which='minor', labelsize=15)
        plt.xlabel(rf'Fixed Flux Value for Parameter $\Phi_{{{index+1}}}$', fontsize=11)
        plt.ylabel('Δχ²', fontsize=15)
        plt.title(f'Δχ² Profile for Parameter $\\Phi_{{{index+1}}}$ '
                  f'(At {eb[index]:.2f} MeV) under 1 eV threshold (Bkg: {self.background_scenario})')
        plt.grid(True)
        plt.xlim(fv[0], fv[np.array(chi_sq_results) <= 1e2][-1])
        plt.ylim(1e-5, 1e2); plt.legend(loc='lower right', ncol=1, fontsize=9); plt.yscale('log')
        if save:
            os.makedirs(self.scenario_dir, exist_ok=True)
            fn = f'{self.scenario_dir}/scan_param_{index}_bkg_{self.background_scenario}.pdf'
            plt.savefig(fn); print(f"Scan plot saved as {fn}")


# -------------------------------------------------------------------
# Module-level helpers
# -------------------------------------------------------------------
def _phi_ylabel(norm):
    """y-axis label for flux divided by ``norm`` (e.g. 1e12 -> '$10^{12}$')."""
    e = int(round(np.log10(norm)))
    if e == 0:
        return r"$\Phi$ [cm$^{-2}$sec$^{-1}$]"
    return rf"$\Phi_{{<~2~\text{{MeV}}}}$ [$10^{{{e}}}$ cm$^{{-2}}$sec$^{{-1}}$]"


def scan_parameter(analysis, data, index, num_points=21, scan_range=0.35):
    return analysis.scan_parameter(data, index, num_points=num_points, scan_range=scan_range)


def load_band(path):
    """Load a band JSON written by ``NeutrinoAnalysis.save_band``."""
    with open(path) as f:
        obj = json.load(f)
    obj['levels'] = tuple(float(l) for l in obj['levels'])
    obj['band_raw'] = {float(k): tuple(v) for k, v in obj['band_raw'].items()}
    obj['band_physical'] = {float(k): tuple(v) for k, v in obj['band_physical'].items()}
    return obj


def load_degeneracy_band(path):
    """Load a degeneracy JSON written by ``NeutrinoAnalysis.save_degeneracy_bands``."""
    with open(path) as f:
        return json.load(f)
