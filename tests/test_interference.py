from pathlib import Path
import numpy as np
import pytest
from eis.model import impedance
from eis.types import Spectrum
from eis.interference import detect_interference
from eis.window_detection import analyze


def make(f,z):
    return Spectrum(Path('noise.txt'),f,z,np.ones(len(f),bool))


@pytest.mark.parametrize('n',[.7,.85,1.0])
@pytest.mark.parametrize('peak',[30,300,3000])
def test_smooth_arc_not_interference(n,peak):
    f = np.logspace(5,-2,75)
    q = 1/(240*(2*np.pi*peak)**n)
    z = impedance(f,12,240,q,n)
    assert len(detect_interference(make(f,z)).excluded)==0


def test_smooth_warburg_not_interference():
    f = np.logspace(5,-2,75)
    w = 25*(1-1j)/np.sqrt(2*np.pi*f)
    z = 12+1/(1/(240+w)+2e-4*(1j*2*np.pi*f)**.85)
    assert len(detect_interference(make(f,z)).excluded)==0


def test_disturbed_band_does_not_change_endpoint_or_parameters():
    f = np.logspace(5,-2,75)
    clean = impedance(f,12,240,2e-4,.85)
    band = (f>=30)&(f<=300)
    area = np.flatnonzero((f>=30)&(f<=150))
    tangent = np.gradient(clean,np.log(f))
    normal = 1j*tangent/abs(tangent)
    normal = np.where(normal.imag>0,-normal,normal)
    results = []
    for amplitude in (.15,.22):
        z = clean.copy()
        z[area] += amplitude*abs(z[area])*normal[area]
        result = analyze(make(f,z))
        assert set(result.noise_indices)==set(area)
        assert np.any(np.isin(result.used,np.flatnonzero(band)))
        assert result.fit.rs==pytest.approx(12,rel=.01)
        assert result.fit.rp==pytest.approx(240,rel=.01)
        assert any('сетевые' in reason for reason in result.reasons)
        results.append(result)
    assert results[0].used.tolist()==results[1].used.tolist()
    assert results[0].fit.rs==pytest.approx(results[1].fit.rs,rel=1e-9)
    assert results[0].fit.rp==pytest.approx(results[1].fit.rp,rel=1e-9)


def test_single_large_off_curve_point_is_removed_without_masking_neighbors():
    f = np.logspace(5,-2,75)
    z = impedance(f,12,240,2e-4,.85)
    index = np.argmin(abs(f-50))
    z[index] -= .4j*abs(z[index])
    r = analyze(make(f,z))
    assert r.noise_indices.tolist()==[index]
    assert index-1 in r.used and index+1 in r.used


def test_moderate_off_curve_point_removed_and_small_deviation_kept():
    f = np.logspace(5,-2,75)
    clean = impedance(f,12,240,2e-4,.85)
    tangent = np.gradient(clean,np.log(f))
    normal = 1j*tangent/abs(tangent)
    strong = np.argmin(abs(f-50))
    small = np.argmin(abs(f-150))
    anchor = np.argmin(abs(f-100))
    z = clean.copy()
    z[strong] += .025*abs(clean[strong])*normal[strong]
    z[small] += .008*abs(clean[small])*normal[small]
    # A clear outlier establishes a local scatter episode; tighten the model
    # distance check within that episode without classifying smooth curvature.
    z[anchor] += .12*abs(clean[anchor])*normal[anchor]
    r = analyze(make(f,z))
    assert r.noise_indices.tolist()==sorted([strong,anchor])
    assert small in r.used
    assert strong-1 in r.used and strong+1 in r.used
    assert r.fit.rp==pytest.approx(240,rel=.01)


@pytest.mark.parametrize('manual',[False,True])
def test_interference_exclusion_stays_within_30_300_hz(manual):
    f = np.sort(np.r_[np.logspace(5,-2,75),29.9,30.,300.,300.1])[::-1]
    z = impedance(f,12,240,2e-4,.85)
    disturbed = np.flatnonzero(np.isin(f,[29.9,30.,300.,300.1]))
    z[disturbed] -= .2j*abs(z[disturbed])
    r = detect_interference(make(f,z),(0,len(f)-1) if manual else None)
    excluded_f = f[r.excluded]
    assert set(excluded_f)=={30.,300.}
