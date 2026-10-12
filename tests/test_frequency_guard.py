from pathlib import Path
import numpy as np
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
