import numpy as np
import pytest
from eis.derived import arc_characteristics
from eis.model import impedance
from eis.fitter import fit_arc


def test_ideal_rc_capacitance_and_peak():
    c,f,c_se,f_se = arc_characteristics([12,240,2e-4,1])
    assert c==pytest.approx(2e-4)
    assert f==pytest.approx(1/(2*np.pi*240*2e-4))
    assert c_se is None and f_se is None


@pytest.mark.parametrize('n',[.6,.85,1.0])
def test_peak_is_model_imaginary_maximum(n):
    params = [12,240,2e-4,n]
    c,f,_,_ = arc_characteristics(params)
    grid = np.geomspace(f/10,f*10,10001)
    assert grid[np.argmax(-impedance(grid,*params).imag)]==pytest.approx(f,rel=.001)
    assert c==pytest.approx(params[2]*(2*np.pi*f)**(n-1))
    assert 2*np.pi*f*params[1]*c==pytest.approx(1)


def test_propagated_se_uses_parameter_correlations():
    params = np.array([12.,240.,2e-4,.85])
    se = np.array([.1,1.,1e-6,.005])
    cov = np.diag(se**2)
    cov[1,2] = cov[2,1] = .5*se[1]*se[2]
    c,f,c_se,f_se = arc_characteristics(params,cov)
    jacobian = np.zeros((2,4))
    for i in range(4):
        step = abs(params[i])*1e-5
        a,b = params.copy(),params.copy()
        a[i]+=step
        b[i]-=step
        jacobian[:,i] = (np.array(arc_characteristics(a)[:2])-np.array(arc_characteristics(b)[:2]))/(2*step)
    expected = np.sqrt(np.diag(jacobian@cov@jacobian.T))
    assert [c_se,f_se]==pytest.approx(expected,rel=1e-5)
    diagonal = arc_characteristics(params,np.diag(se**2))
    assert not np.isclose(f_se,diagonal[3],rtol=.001)


def test_actual_fit_supplies_full_covariance_for_derived_se():
    f = np.logspace(5,-2,75)
    z = impedance(f,12,240,2e-4,.85)
    rng = np.random.default_rng(5)
    z += .001*abs(z)*(rng.normal(size=len(f))+1j*rng.normal(size=len(f)))
    fit = fit_arc(f,z)
    assert fit.covariance.shape==(4,4)
    assert np.sqrt(np.diag(fit.covariance))==pytest.approx(fit.se)
    assert fit.c_eff_se>0 and fit.f_max_se>0


def test_nonrepresentable_values_are_na_not_inf_or_zero():
    assert arc_characteristics([12,240,2e-4,1e-6])[:2]==(None,None)
