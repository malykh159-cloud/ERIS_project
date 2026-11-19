import numpy as np
from scipy.optimize import minimize_scalar

TRANSMISSIVE_NAME = 'Rs + [CPE ‖ (Rp + Z₀)] (test)'
BOUNDARY_NAME = 'Rs + [CPE ‖ (Rp + Z_D)] (с граничным CPE)'
TRANSMISSIVE_PARAMETERS = (
    ('Rs, Ω', 'Rs', 'Ohm'), ('Rp, Ω', 'Rp', 'Ohm'), ('Q, S·sⁿ', 'Q', 'S_s_n'),
    ('n', 'n', ''), ('B, √с', 'B', 'sqrt_s'), ('R_D, Ω', 'R_D', 'Ohm'),
)
BOUNDARY_PARAMETERS = (
    ('Rs, Ω', 'Rs', 'Ohm'), ('R, Ω', 'R', 'Ohm'), ('Qdl, S·sⁿ', 'Qdl', 'S_s_n'),
    ('n', 'n', ''), ('R_D, Ω', 'R_D', 'Ohm'), ('τ_D, с', 'tau_D', 's'),
    ('QT, S·sᵐ', 'QT', 'S_s_m'), ('m', 'm', ''),
)


def diffusion_line(s, rd, tau_d, y_t):
    u_squared = s * tau_d
    u = np.sqrt(u_squared)
    small = np.abs(u) < 1e-3
    tanh_u = np.tanh(u, out=np.ones_like(u), where=u.real < 20)
    a = np.divide(tanh_u, u, out=np.ones_like(u), where=~small)
    a = np.where(small, 1 - u_squared / 3 + 2 * u_squared**2 / 15, a)
    return (1 + rd * a * y_t) / (y_t + u_squared / rd * a)


def boundary_impedance(f, rs, r, q_dl, n, rd, tau_d, q_t, m):
    """Randles circuit with finite diffusion terminated by a CPE, f > 0."""
    s = 2j * np.pi * np.asarray(f)
    zd = diffusion_line(s, rd, tau_d, q_t * s**m)
    return rs + 1 / (q_dl * s**n + 1 / (r + zd))


def impedance(f, rs, rp, q, n, b, rd):
    jw = 2j * np.pi * np.asarray(f)
    u = b * np.sqrt(jw)
    # For Re(u) >= 20, tanh(u) is 1 to floating point precision.
    tanh_u = np.tanh(u, out=np.ones_like(u), where=u.real < 20)
    zd = rd * np.divide(tanh_u, u, out=np.ones_like(u), where=u != 0)
    return rs + 1 / (q * jw**n + 1 / (rp + zd))


def peak_characteristics(params, f_min, f_max):
    """Return CPE effective capacitance and the full model's in-band peak."""
    if f_min >= f_max:
        return None, None
    grid = np.linspace(np.log(f_min), np.log(f_max), 1025)
    height = -impedance(np.exp(grid), *params).imag
    if not np.all(np.isfinite(height)):
        return None, None
    peaks = np.flatnonzero((height[1:-1] > height[:-2]) &
                           (height[1:-1] >= height[2:])) + 1
    if not len(peaks):
        return None, None
    candidates = [
        minimize_scalar(lambda x: impedance(np.exp(x), *params).imag,
                        bounds=(grid[i-1], grid[i+1]), method='bounded')
        for i in peaks
    ]
    best = min(candidates, key=lambda p: p.fun)
    tolerance = 1e-8 * max(abs(best.fun), np.finfo(float).tiny)
    if (not best.success or -best.fun <= max(height[0], height[-1]) + tolerance
            or any(p is not best and p.fun - best.fun <= tolerance for p in candidates)):
        return None, None
    frequency = float(np.exp(best.x))
    return float(params[2] * (2 * np.pi * frequency)**(params[3] - 1)), frequency
