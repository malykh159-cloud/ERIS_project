"""Look for sustained recovery of Nyquist slope toward a low-frequency tail."""
import numpy as np
from scipy.ndimage import median_filter
from . import config as C


def tail_candidate(frequency,z):
    if len(z) < C.MIN_POINTS+C.CONFIRM_POINTS:
        return None
    x,y = z.real,-z.imag
    dx = np.diff(x)
    span = max(float(np.ptp(x)),np.finfo(float).tiny)
    valid = dx > C.GEOMETRY_MIN_DX_REL*span
    slope = np.divide(np.diff(y),dx,out=np.full_like(dx,np.nan),where=valid)
    # Missing/nonmonotone dx cannot establish a physical tail.
    smooth = median_filter(np.where(valid,slope,0.0),size=C.GEOMETRY_SMOOTH_SIZE,mode='nearest')
    minimum,minimum_edge,streak = np.inf,None,0
    amplitude = C.TAIL_MIN_RISE_REL*max(float(np.ptp(y)),np.finfo(float).tiny)
    for edge in range(len(dx)):
        if not valid[edge] or frequency[edge] > C.AUTO_BOUNDARY_MAX_HZ:
            streak = 0
            continue
        if smooth[edge] < minimum:
            minimum,minimum_edge = smooth[edge],edge
            streak = 0
        if minimum_edge is None:
            continue
        rising = smooth[edge] > minimum+C.TAIL_SLOPE_REBOUND
        meaningful = y[edge+1]-y[minimum_edge+1] > amplitude
        streak = streak+1 if rising and meaningful else 0
        if streak >= C.CONFIRM_POINTS:
            # Include the points up to the minimum-slope edge. Never select above
            # the 2 kHz guard, and never use an isolated peak as an endpoint.
            endpoint = minimum_edge+2
            return endpoint if endpoint >= C.MIN_POINTS else None
    return None
