"""Peak-frequency (Hsu-Mansfeld) effective capacitance of the fitted Rp || CPE."""
import numpy as np


def positive_exp(value):
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        result = float(np.exp(value))
    return result if np.isfinite(result) and result>0 else None


def arc_characteristics(params,covariance=None):
    """Return C_eff [F], f_max [Hz], and their propagated SE using full covariance."""
    _,rp,q,n = np.asarray(params,float)
    if not np.all(np.isfinite([rp,q,n])) or min(rp,q,n)<=0:
        return None,None,None,None
    combined = np.log(rp)+np.log(q)
    log_omega = -combined/n
    capacitance = positive_exp(-np.log(rp)-log_omega)
    peak = positive_exp(log_omega-np.log(2*np.pi))

    def standard_error(value,log_gradient):
        if value is None or covariance is None:
            return None
        gradient = value*np.asarray(log_gradient)
        with np.errstate(over='ignore',invalid='ignore'):
            variance = float(gradient@covariance@gradient)
        if not np.isfinite(variance) or variance<0:
            return None
        return float(np.sqrt(variance))

    c_gradient = [0,(1/n-1)/rp,1/(n*q),-combined/n**2]
    f_gradient = [0,-1/(n*rp),-1/(n*q),combined/n**2]
    return capacitance,peak,standard_error(capacitance,c_gradient),standard_error(peak,f_gradient)
