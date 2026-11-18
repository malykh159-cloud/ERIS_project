"""Find a sustained reversal of the Nyquist tangent on the descending side."""
import numpy as np
from scipy.ndimage import median_filter
from . import config as C


def tail_candidate(frequency,z):
    return next(tail_candidates(frequency, z), None)


def tail_candidates(frequency, z, *, allow_shoulder=False):
    if len(z) < C.MIN_POINTS+C.CONFIRM_POINTS:
        return
    x,y = z.real,-z.imag
    dx, dy = np.diff(x), np.diff(y)
    span = max(float(np.ptp(x)),np.finfo(float).tiny)
    valid = dx > C.GEOMETRY_MIN_DX_REL * span
    # Smooth local tangents in frequency order. Angles keep nearly vertical
    # edges bounded as the arc approaches its real-axis intercept.
    angle = np.arctan2(dy, dx)
    smooth = median_filter(angle, size=C.GEOMETRY_SMOOTH_SIZE, mode='nearest')
    minimum,minimum_edge,streak = (np.inf if allow_shoulder else 0.0),None,0
    for edge in range(len(dx)):
        if not valid[edge] or frequency[edge] > C.AUTO_BOUNDARY_MAX_HZ:
            streak = 0
            continue
        if smooth[edge] < minimum:
            minimum, minimum_edge = smooth[edge], edge
            streak = 0
        if minimum_edge is None:
            continue
        rising = smooth[edge] > minimum + C.TAIL_TANGENT_REBOUND_RAD
        streak = streak + 1 if rising else 0
        if streak == C.CONFIRM_POINTS:
            # Include the points up to the minimum-slope edge. Never select above
            # the 2 kHz guard, and never use an isolated peak as an endpoint.
            endpoint = minimum_edge+2
            if endpoint >= C.MIN_POINTS:
                yield endpoint
