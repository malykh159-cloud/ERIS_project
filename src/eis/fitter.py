import numpy as np
from scipy.optimize import least_squares
from .model import impedance
from .types import Fit
from . import config as C


def fit_arc(frequency, z, seed=None, robust_scale=None):
    f, z = np.asarray(frequency, float), np.asarray(z, complex)
    if len(f) < C.MIN_POINTS:
        raise ValueError(f"Для фитирования требуется минимум {C.MIN_POINTS} точек.")
    # Log Rs/Rp/Q enforce positivity and keep Jacobian columns comparable.
    scale = np.maximum(np.abs(z), np.finfo(float).tiny)

    def unpack(t):
        return np.r_[np.exp(t[:3]), t[3]]

    def residual(t):
        dz = (z - impedance(f, *unpack(t))) / scale
        return np.r_[dz.real, dz.imag]

    rs = max(float(np.min(z.real)) * 0.8, float(np.max(scale)) * 1e-6)
    rp = max(float(np.ptp(z.real)), 2 * float(np.max(-z.imag)), rs * 0.1)
    peak = f[np.argmax(-z.imag)]
    starts = []
    if seed is not None:
        starts.append(np.r_[np.log(seed[:3]), seed[3]])
    for n in C.START_N:
        for factor in C.START_RP_FACTORS:
            r = rp * factor
            q = 1 / (r * (2 * np.pi * peak) ** n)
            starts.append(np.r_[np.log([rs, r, q]), n])
    lower = np.array([C.LOG_PARAMETER_MIN] * 3 + [C.N_MIN])
    upper = np.array([C.LOG_PARAMETER_MAX] * 3 + [1.0])
    solutions = []
    for start in starts:
        sol = least_squares(residual, np.clip(start, lower + 1e-8, upper - 1e-8),
                            bounds=(lower, upper), x_scale='jac', max_nfev=C.MAX_EVALUATIONS,
                            loss='linear' if robust_scale is None else 'soft_l1',
                            f_scale=1.0 if robust_scale is None else robust_scale,
                            ftol=C.OPT_TOL, xtol=C.OPT_TOL, gtol=C.OPT_TOL)
        if sol.success and np.all(np.isfinite(sol.fun)):
            solutions.append(sol)
    if not solutions:
        raise ValueError("Не удалось получить устойчивое решение CNLS.")
    best = min(solutions, key=lambda s: s.cost)
    params = unpack(best.x)
    chi2 = float(np.dot(best.fun, best.fun) / (2 * len(f) - C.PARAMETERS))
    _, singular, vt = np.linalg.svd(best.jac, full_matrices=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
    se = None
    covariance = None
    if robust_scale is None and condition < C.MAX_JAC_CONDITION:
        cov_log = (vt.T * (chi2 / singular**2)) @ vt
        # Delta-method change of coordinates for positive physical parameters.
        transform = np.r_[params[:3], 1.0]
        errors = np.sqrt(np.maximum(np.diag(cov_log), 0)) * transform
        if np.all(np.isfinite(errors)):
            se = errors
            covariance = cov_log*np.outer(transform,transform)
    boundary = bool(np.any(best.x - lower < C.BOUND_MARGIN) or
                    np.any(upper - best.x < C.BOUND_MARGIN))
    near = [s for s in solutions if s.cost <= best.cost * C.NEAR_OPTIMUM_COST_RATIO + C.NUMERICAL_COST_FLOOR]
    stable = all(abs(np.exp(s.x[1]) / params[1] - 1) < C.MAX_RP_DRIFT for s in near)
    dz = (z - impedance(f, *params)) / scale
    return Fit(params, se, chi2, dz, condition, boundary, stable,covariance)
