from pathlib import Path
import numpy as np
import pytest
from eis.model import impedance
from eis.types import Spectrum
from eis.window_detection import analyze
from eis.geometry import tail_candidate


def make(f,z):
    return Spectrum(Path('guard.txt'),f,z,np.ones(len(f),bool))


def test_early_high_frequency_transition_cannot_end_above_2000():
    f = np.logspace(6,0,70)
    q1 = 1/(120*(2*np.pi*20000)**.85)
    q2 = 1/(200*(2*np.pi*50)**.85)
    z = impedance(f,12,120,q1,.85)+impedance(f,0,200,q2,.85)
    result = analyze(make(f,z))
    assert result.fit is not None
    assert f[result.used[-1]]<=2000


def test_no_data_below_guard_keeps_all_and_warns():
    f = np.logspace(5,3.5,20)
    z = impedance(f,12,240,2e-4,.85)
    result = analyze(make(f,z))
    assert len(result.used)==len(f)
    assert result.status=='Проверьте диапазон'
    assert any('не достигает 2000' in reason for reason in result.reasons)


def test_manual_can_end_above_guard():
    f = np.logspace(5,1,50)
    result = analyze(make(f,impedance(f,12,240,2e-4,.85)),(0,12))
    assert result.fit is not None
    assert f[result.used[-1]]>2000
    assert result.manual


def test_apex_is_not_end_of_clean_arc():
    f = np.logspace(5,-2,75)
    z = impedance(f,12,240,2e-4,.85)
    assert tail_candidate(f,z) is None


@pytest.mark.parametrize('manual', [None, (0, 20)])
def test_100khz_cutoff_preserves_raw_points_and_fit(manual):
    f = np.r_[1e6, 100000.1, 100000., np.geomspace(50000, 1, 18)]
    z = impedance(f, 12, 240, 2e-4, .85)
    z[:2] = [1e8-1e8j, 1e7-1e7j]
    s = make(f, z)
    before = s.z.copy()
    result = analyze(s, manual)
    assert result.fit is not None
    assert not np.any(np.isin(result.used, [0, 1]))
    assert 2 in result.used  # Exactly 100 kHz is retained.
    assert result.fit.rs == pytest.approx(12, rel=1e-5)
    assert result.fit.rp == pytest.approx(240, rel=1e-5)
    assert np.array_equal(s.z, before)
    assert len(s.frequency) == len(f)
