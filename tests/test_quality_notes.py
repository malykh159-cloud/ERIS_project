from pathlib import Path
import runpy
import numpy as np
import pytest
from eis.fitter import fit_arc
from eis.model import impedance
from eis.quality import assess
from eis.types import Spectrum
from eis.window_detection import analyze


def test_small_correlated_residuals_are_notes_but_large_residuals_warn():
    f = np.geomspace(1e5, 1, 35)
    z = impedance(f, 12, 240, 2e-4, .85)
    fit = fit_arc(f, z)
    fit.residual = np.linspace(-.001, .001, len(f)).astype(complex)
    notes = []
    assert assess(fit, f, z, notes=notes) == []
    assert any('коррелированы' in s for s in notes)
    fit.chi2 = .001
    assert any('повышенные' in s for s in assess(fit, f, z, notes=[]))


def test_short_valid_fit_gets_note_and_bad_uncertainty_still_warns():
    f = np.geomspace(1000, .1, 10)
    z = impedance(f, 12, 240, 2e-4, .85)
    fit = fit_arc(f, z)
    notes = []
    assert assess(fit, f, z, notes=notes) == []
    assert any('12 точек' in s for s in notes)
    fit.se = None
    assert any('Ковариация' in s for s in assess(fit, f, z, notes=[]))


def test_series_l_diagnostic_recovers_known_inductance_and_rejects_clean_arc():
    compare = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                 'tools/check_inductance.py'))['compare_series_l']
    f = np.geomspace(1e5, 1, 40)
    params = np.array([12, 240, 2e-4, .85])
    z = impedance(f, *params)
    contaminated = z + 1j * 2 * np.pi * f * 2e-5
    before = contaminated.copy()
    report = compare(f, contaminated, params)
    assert report['inductance_H'] == pytest.approx(2e-5, rel=1e-4)
    assert report['relative_sse_improvement'] > .99
    assert report['series_l_params'] == pytest.approx(params, rel=1e-4)
    assert np.array_equal(before, contaminated)
    assert compare(f, z, params)['inductance_H'] < 1e-9


def test_geometric_tail_is_note_when_numeric_checks_pass():
    # Match a smooth transition whose CNLS jump is too gradual for the streak test.
    f = np.geomspace(1e5, .01, 75)
    z = impedance(f, 12, 240, 2e-4, .85)
    z[48:] += 80 * (np.sqrt(f[48] / f[48:]) - .8) * (1 - 1j)
    s = Spectrum(Path('tail.txt'), f, z, np.ones(len(f), bool))
    result = analyze(s)
    assert len(result.used) < len(f)
    assert any('хвост' in s for s in result.notes)
    assert not any('граница' in s for s in result.reasons)
    assert result.status == 'OK', result.reasons
