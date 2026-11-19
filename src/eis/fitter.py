import numpy as np
from scipy.optimize import least_squares
from .model import impedance
from .types import BoundaryFit, Fit, SangbayFit
from . import sangbay_model
from . import config as C


def fit_arc(frequency, z, seed=None, robust_scale=None, *, sangbay=False, boundary_cpe=False,
            cancelled=lambda: False):
    f, z = np.asarray(frequency, float), np.asarray(z, complex)
    if len(f) < C.MIN_POINTS:
        raise ValueError(f"Для фитирования требуется минимум {C.MIN_POINTS} точек.")
    # Log Rs/Rp/Q enforce positivity and keep Jacobian columns comparable.
    scale = np.maximum(np.abs(z), np.finfo(float).tiny)

    if boundary_cpe:
        model = sangbay_model.boundary_impedance
        positive = [2, 4, 5, 6]  # Rs and R can reach zero.
    else:
        model = sangbay_model.impedance if sangbay else impedance
        positive = [0, 1, 2, 4, 5] if sangbay else [0, 1, 2]

    def unpack(t):
        params = t.copy()
        params[positive] = np.exp(t[positive])
        return params

    def residual(t):
        if cancelled():
            raise InterruptedError()
        dz = (z - model(f, *unpack(t))) / scale
        return np.concatenate((dz.real, dz.imag))

    rs = max(float(np.min(z.real)) * 0.8, float(np.max(scale)) * 1e-6)
    rp = max(float(np.ptp(z.real)), 2 * float(np.max(-z.imag)), rs * 0.1)
    peak = f[np.argmax(-z.imag)]
    starts = []
    if seed is not None:
        start = seed.copy()
        start[positive] = np.log(start[positive])
        starts.append(start)
    if boundary_cpe:
        # Documented reference start plus data-scaled starts spanning the band.
        starts.append(np.array([6, 9, np.log(5e-4), .56, np.log(28), np.log(.094), np.log(.0057), .70]))
        high = np.argsort(f)[-min(4, len(f)):]
        rs = max(float(np.min(z.real[high])) * 0.9, 0.0)
        r_scale = max(float(np.median(np.abs(z))), 1e-8)
        low = np.argmin(f)
        for tau_frequency in np.geomspace(min(f), max(f), 6)[:4]:
            for n, m in ((.55, .7), (.88, .45), (.7, .95)):
                tau = 1 / (2 * np.pi * tau_frequency)
                q_dl = 1 / (r_scale * (2 * np.pi * np.median(f))**n)
                q_t = 1 / (scale[low] * (2 * np.pi * f[low])**m)
                starts.append(np.array([rs, r_scale / 2, np.log(q_dl), n, np.log(r_scale),
                                        np.log(tau), np.log(q_t), m]))
    for n in C.START_N:
        if boundary_cpe:
            break
        for factor in C.START_RP_FACTORS:
            r = rp * factor
            q = 1 / (r * (2 * np.pi * peak) ** n)
            if sangbay:
                for diffusion_frequency in np.geomspace(min(f), max(f), 5)[:3]:
                    b = 1 / np.sqrt(2 * np.pi * diffusion_frequency)
                    starts.append(np.r_[np.log([rs, r/2, q]), n, np.log([b, r/2])])
            else:
                starts.append(np.r_[np.log([rs, r, q]), n])
    extra = 2 if sangbay else 0
    lower = np.array([C.LOG_PARAMETER_MIN] * 3 + [C.N_MIN] + [C.LOG_PARAMETER_MIN] * extra)
    upper = np.array([C.LOG_PARAMETER_MAX] * 3 + [1.0] + [C.LOG_PARAMETER_MAX] * extra)
    if boundary_cpe:
        lo, hi = C.LOG_PARAMETER_MIN, C.LOG_PARAMETER_MAX
        lower = np.array([0, 0, lo, C.N_MIN, lo, lo, lo, C.N_MIN])
        upper = np.array([np.inf, np.inf, hi, 1, hi, hi, hi, 1])
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
    chi2 = float(np.dot(best.fun, best.fun) / (2 * len(f) - len(params)))
    _, singular, vt = np.linalg.svd(best.jac, full_matrices=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
    se = None
    covariance = None
    if robust_scale is None and condition < C.MAX_JAC_CONDITION:
        cov_log = (vt.T * (chi2 / singular**2)) @ vt
        # Delta-method change of coordinates for positive physical parameters.
        transform = np.ones_like(params)
        transform[positive] = params[positive]
        errors = np.sqrt(np.maximum(np.diag(cov_log), 0)) * transform
        if np.all(np.isfinite(errors)):
            se = errors
            covariance = cov_log*np.outer(transform,transform)
    boundary = bool(np.any(best.x - lower < C.BOUND_MARGIN) or
                    np.any(upper - best.x < C.BOUND_MARGIN))
    near = [s for s in solutions if s.cost <= best.cost * C.NEAR_OPTIMUM_COST_RATIO + C.NUMERICAL_COST_FLOOR]
    checked = slice(None) if sangbay or boundary_cpe else [1]
    denominator = np.maximum(abs(params[checked]), np.finfo(float).tiny)
    if boundary_cpe:
        denominator[[0, 1]] = np.maximum(denominator[[0, 1]], scale.min() * 1e-8)
    stable = all(np.all(abs(unpack(s.x)[checked] - params[checked]) / denominator < C.MAX_RP_DRIFT)
                 for s in near)
    dz = (z - model(f, *params)) / scale
    if boundary_cpe:
        return BoundaryFit(params, se, chi2, dz, condition, boundary, stable, covariance)
    if sangbay:
        return SangbayFit(params, se, chi2, dz, condition, boundary, stable, covariance,
                          (float(min(f)), float(max(f))))
    return Fit(params, se, chi2, dz, condition, boundary, stable,covariance)
