from pathlib import Path

import numpy as np
import pytest

from eis.sangbay_model import boundary_impedance, diffusion_line, impedance, peak_characteristics
from eis.types import Spectrum
from eis.window_detection import analyze


def test_sangbay_recovers_diffusion_and_keeps_low_frequencies():
    f = np.r_[150_000, np.geomspace(100_000, 0.001, 81)]
    params = [7.0, 25.0, 0.004, 0.86, 0.9, 45.0]
    z = impedance(f, *params)
    spectrum = Spectrum(Path('synthetic.txt'), f, z, np.ones(len(f), dtype=bool))
    result = analyze(spectrum, sangbay=True)

    assert result.fit is not None, result.reasons
    np.testing.assert_array_equal(result.used, np.arange(1, len(f)))
    np.testing.assert_allclose(result.fit.params, params, rtol=1e-5)
    assert impedance(0, *params) == pytest.approx(77.0)
    assert result.fit.f_max is not None
    assert result.fit.c_eff == pytest.approx(params[2] * (2*np.pi*result.fit.f_max)**(params[3]-1))
    assert peak_characteristics(params, 1_000, 100_000) == (None, None)


REFERENCE_PARAMETERS = [6, 9, 5e-4, .56, 28, .094, 5.7e-3, .70]


def test_boundary_cpe_matches_documented_impedance():
    f = [0.01, 1, 1000, 100000]
    expected = [386.491054363627-823.703604000145j, 34.241362253933-26.395751990425j,
                11.920187604827-2.469073691778j, 6.733796630212-.742953424910j]
    np.testing.assert_allclose(boundary_impedance(f, *REFERENCE_PARAMETERS), expected,
                               rtol=1e-8, atol=1e-8)

    for q_t, limit in [(1e-12, 24.099023300750-44.351029409876j),
                       (1e12, 40.273252217782-6.149504442030j)]:
        params = REFERENCE_PARAMETERS.copy()
        params[6] = q_t
        assert boundary_impedance(1, *params) == pytest.approx(limit, rel=1e-8)


def test_diffusion_line_matches_loaded_transmission_line_and_limits():
    rd, tau, q_t, m = REFERENCE_PARAMETERS[4:]
    s = 2j * np.pi * np.geomspace(1e-9, 10, 20)
    u = np.sqrt(s * tau)
    z0, zt = rd / u, 1 / (q_t * s**m)
    direct = z0 * (zt + z0 * np.tanh(u)) / (z0 + zt * np.tanh(u))
    np.testing.assert_allclose(diffusion_line(s, rd, tau, q_t * s**m), direct, rtol=1e-8)
    low = 2j * np.pi * 1e-12
    assert diffusion_line(low, rd, tau, q_t) == pytest.approx(rd + 1/q_t, rel=1e-8)
    assert diffusion_line(low, rd, tau, q_t*low) == pytest.approx(1/(low*(q_t+tau/rd)), rel=1e-8)
    high = 2j * np.pi * 1e8
    assert diffusion_line(high, rd, tau, q_t*high**m) == pytest.approx(rd/np.sqrt(high*tau), rel=1e-8)


def test_boundary_cpe_fits_full_synthetic_spectrum():
    f = np.r_[150_000, np.geomspace(100_000, 0.01, 71)]
    z = boundary_impedance(f, *REFERENCE_PARAMETERS)
    result = analyze(Spectrum(Path('synthetic.txt'), f, z, np.ones(len(f), dtype=bool)),
                     boundary_cpe=True)
    assert result.boundary_cpe and result.fit is not None, result.reasons
    np.testing.assert_array_equal(result.used, np.arange(1, len(f)))
    np.testing.assert_allclose(result.fit.params, REFERENCE_PARAMETERS, rtol=1e-5)
    assert result.fit.rms_percent < 1e-6
