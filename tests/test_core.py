from pathlib import Path
import numpy as np
import pytest
from eis.model import impedance
from eis.fitter import fit_arc
from eis.types import Spectrum
from eis.window_detection import analyze


def spectrum(f, z):
    return Spectrum(Path('synthetic.txt'), f, z, np.ones(len(f), bool))


def sample(count=60, low=-2, noise=0):
    f = np.logspace(5, low, count)
    z = impedance(f, 12, 240, 2e-4, 0.85)
    rng = np.random.default_rng(42)
    z = z + noise * abs(z) * (rng.normal(size=count) + 1j * rng.normal(size=count))
    return f, z


def test_model_limits():
    z = impedance([1e30, 1e-30], 12, 240, 2e-4, 0.85)
    assert z.real == pytest.approx([12, 252])


def test_clean_arc():
    f, z = sample(noise=0.001)
    r = analyze(spectrum(f, z))
    assert r.fit.rs == pytest.approx(12, rel=0.02)
    assert r.fit.rp == pytest.approx(240, rel=0.02)
    assert len(r.used) == len(f)
    assert r.status == 'OK', r.reasons


def test_tail():
    f, z = sample(count=75, noise=0.0005)
    onset = 48
    z[onset:] += 80 * (np.sqrt(f[onset] / f[onset:]) - 0.8) * (1 - 1j)
    r = analyze(spectrum(f, z))
    assert onset - 4 <= len(r.used) <= onset + 2, (len(r.used), r.reasons)
    assert r.fit.rs == pytest.approx(12, rel=0.03)
    assert r.fit.rp == pytest.approx(240, rel=0.06)


def test_incomplete_uncertainty():
    f, z = sample(count=90, noise=0.002)
    full = fit_arc(f, z)
    cut = fit_arc(f[:50], z[:50])
    assert cut.rp == pytest.approx(240, rel=0.1)
    assert cut.se[1] > full.se[1]


def test_inductive_exclusion():
    f, z = sample()
    z[:4] = z[:4].real + 2j
    r = analyze(spectrum(f, z))
    assert not any(i in r.used for i in range(4))
    assert r.fit.rs == pytest.approx(12, rel=0.01)


def test_two_arcs_warn():
    f, z = sample()
    z += impedance(f, 0, 150, 0.008, 0.8)
    r = analyze(spectrum(f, z))
    assert r.status != 'OK'


def test_single_outlier_not_boundary():
    f, z = sample(noise=0.001)
    z[35] *= 1.03
    r = analyze(spectrum(f, z))
    assert len(r.used) == len(f)


def test_unstable_terminal_cluster_below_interference_band():
    f, z = sample(count=75)
    z[-7:] += np.array([.17, -.35, .4, -.44, .5, -.4, .27]) * abs(z[-7:])
    s = spectrum(f, z)
    before = z.copy()
    result = analyze(s)
    assert result.used.tolist() == list(range(len(f) - 7))
    assert result.unstable_indices.tolist() == list(range(len(f) - 7, len(f)))
    assert max(f[result.unstable_indices]) < 30
    assert len(result.noise_indices) == 0
    assert result.fit.rp == pytest.approx(240, rel=0.01)
    assert np.array_equal(s.z, before)

    manual = analyze(s, (0, len(f) - 1))
    assert len(manual.used) == len(f)
    assert len(manual.unstable_indices) == 0


def test_manual_excludes_both_ends():
    f, z = sample()
    r = analyze(spectrum(f, z), (8, 45))
    assert list(r.used) == list(range(8, 46))
    assert r.manual
