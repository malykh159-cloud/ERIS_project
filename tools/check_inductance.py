"""Reproducible diagnostic only: positive series L, without changing raw data or masks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eis.fitter import fit_arc
from eis.model import impedance
from eis.parser import read_spectrum


def compare_series_l(f, z, reference_params=None):
    f, z = np.asarray(f), np.asarray(z)
    base = fit_arc(f, z)
    # Scale L so its last-coordinate sensitivity is comparable to other parameters.
    omega = 2 * np.pi * f
    l_scale = float(np.median(abs(z)) / max(omega))

    def residual(t):
        params = np.r_[np.exp(t[:3]), t[3]]
        dz = (z - impedance(f, *params) - 1j * omega * t[4] * l_scale) / abs(z)
        return np.r_[dz.real, dz.imag]

    seeds = [base.params]
    if reference_params is not None:
        seeds.append(reference_params)
    solutions = []
    for p in seeds:
        for initial_l in (0., 1.):
            sol = least_squares(
                residual, np.r_[np.log(p[:3]), p[3], initial_l],
                bounds=([-50, -50, -50, 1e-6, 0], [50, 50, 50, 1, np.inf]),
                x_scale='jac', max_nfev=4000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if sol.success and np.all(np.isfinite(sol.fun)):
                solutions.append(sol)
    if not solutions:
        raise ValueError('Series-L diagnostic did not converge')
    best = min(solutions, key=lambda s: s.cost)
    base_sse = base.chi2 * (2 * len(f) - 4)
    return {
        'points': len(f), 'base_params': base.params.tolist(), 'base_chi2': base.chi2,
        'series_l_params': np.r_[np.exp(best.x[:3]), best.x[3]].tolist(),
        'inductance_H': float(best.x[4] * l_scale),
        'series_l_chi2': float(2 * best.cost / (2 * len(f) - 5)),
        'relative_sse_improvement': float(1 - 2 * best.cost / max(base_sse, 1e-30)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spectrum', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = hashlib.sha256(args.spectrum.read_bytes()).hexdigest()
    s = read_spectrum(args.spectrum)
    # Match the screenshot's lower endpoint and reference arc, not the LF tail.
    indices = s.eligible[s.frequency[s.eligible] >= 15.849]
    f, z = s.frequency[indices], s.z[indices]
    reference = fit_arc(f[f <= 1000], z[f <= 1000])
    report = compare_series_l(f, z, reference.params)
    hf = f > 1000
    implied = (z - impedance(f, *reference.params)).imag / (2 * np.pi * f)
    report.update({
        'file': str(args.spectrum), 'sha256': before,
        'frequency_range_Hz': [float(max(f)), float(min(f))],
        'reference_range_Hz': [1000., 15.849],
        'reference_params': reference.params.tolist(),
        'reference_chi2': reference.chi2,
        'high_frequency_Hz': f[hf].tolist(),
        'implied_inductance_relative_to_reference_H': implied[hf].tolist(),
        'source_hash_unchanged': before == hashlib.sha256(args.spectrum.read_bytes()).hexdigest(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
