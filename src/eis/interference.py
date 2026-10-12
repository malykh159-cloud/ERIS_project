"""Flag localized irregular scatter, not its physical cause, in the user-specified band."""
from dataclasses import dataclass,field
import numpy as np
from scipy.ndimage import median_filter
from . import config as C
from .fitter import fit_arc
from .model import impedance
from .geometry import tail_candidate


@dataclass
class Interference:
    excluded: np.ndarray = field(default_factory=lambda: np.array([],dtype=int))
    suspects: np.ndarray = field(default_factory=lambda: np.array([],dtype=int))
    maximum_deviation: float = 0.0
    threshold: float = C.INTERFERENCE_REL_FLOOR
    warning: str = ''


def local_scatter(spectrum):
    indices = spectrum.eligible
    if len(indices) < C.INTERFERENCE_TREND_SIZE + 2:
        return Interference()
    f,z = spectrum.frequency[indices],spectrum.z[indices]
    band = (f >= C.INTERFERENCE_LOW_HZ) & (f <= C.INTERFERENCE_HIGH_HZ)
    if np.sum(band) < C.INTERFERENCE_MIN_POINTS:
        return Interference()
    x = np.log(f)
    denominator = x[2:]-x[:-2]
    weight = np.divide(x[1:-1]-x[:-2],denominator,out=np.zeros_like(denominator),where=denominator!=0)
    predicted = (1-weight)*z[:-2]+weight*z[2:]
    local = (z[1:-1]-predicted)/np.maximum(abs(z[1:-1]),np.finfo(float).tiny)
    # Second-order curvature evolves smoothly for a clean arc/tail. Subtract its
    # running median so curvature itself is not classified as mains interference.
    trend = median_filter(local.real,size=C.INTERFERENCE_TREND_SIZE,mode='nearest') + 1j*median_filter(local.imag,size=C.INTERFERENCE_TREND_SIZE,mode='nearest')
    deviation = abs(local-trend)
    valid = denominator != 0
    inside = band[1:-1] & valid
    reference = deviation[(~band[1:-1]) & valid & (f[1:-1] <= C.AUTO_BOUNDARY_MAX_HZ)]
    threshold = C.INTERFERENCE_REL_FLOOR
    if len(reference):
        center = float(np.median(reference))
        mad = C.MAD_SCALE*float(np.median(abs(reference-center)))
        threshold = max(threshold,center+C.INTERFERENCE_MAD_MULTIPLIER*mad)
    suspects = indices[1:-1][inside & (deviation > threshold)]
    peak = float(np.max(deviation[inside])) if np.any(inside) else 0.0
    if len(suspects) < C.INTERFERENCE_MIN_SUSPECTS:
        return Interference(maximum_deviation=peak,threshold=threshold)
    return Interference(indices[band],suspects,peak,threshold)


def curve_distance(f,z,params):
    """Relative shortest distance to a model Nyquist polyline, not to the nearest frequency."""
    curve = impedance(np.geomspace(max(f),min(f),C.INTERFERENCE_CURVE_POINTS),*params)
    start,delta = curve[:-1],np.diff(curve)
    length2 = np.maximum(abs(delta)**2,np.finfo(float).tiny)
    vectors = z[:,None]-start[None,:]
    position = np.clip((vectors.real*delta.real+vectors.imag*delta.imag)/length2,0,1)
    distance = np.min(abs(vectors-position*delta),axis=1)
    return distance/np.maximum(abs(z),np.finfo(float).tiny)


def detect_interference(spectrum,manual=None,cancelled=lambda:False):
    local = local_scatter(spectrum)
    if not len(local.excluded):
        return local
    indices = spectrum.eligible
    if manual is not None:
        a,b = sorted(manual)
        indices = indices[(indices>=a)&(indices<=b)]
    else:
        # The entire band is used ONLY as a provisional reference exclusion,
        # never as the final point mask. It protects the initial shape search.
        quiet = np.setdiff1d(indices,local.excluded)
        end = tail_candidate(spectrum.frequency[quiet],spectrum.z[quiet])
        if end is not None:
            indices = indices[indices<=quiet[end-1]]
    if len(indices)<C.MIN_POINTS:
        return Interference(suspects=local.suspects,warning='Недостаточно точек для проверки наводок по модели; точки сохранены.')
    f,z = spectrum.frequency[indices],spectrum.z[indices]
    band = (f>=C.INTERFERENCE_LOW_HZ)&(f<=C.INTERFERENCE_HIGH_HZ)
    if not np.any(band):
        return Interference(suspects=local.suspects)
    excluded = np.zeros(len(f),dtype=bool)
    seed = None
    threshold = C.INTERFERENCE_MODEL_REL_FLOOR
    try:
        for _ in range(C.INTERFERENCE_MAX_PASSES):
            if cancelled():
                raise InterruptedError('Обработка отменена.')
            keep = ~excluded
            if sum(keep)<C.MIN_POINTS:
                return Interference(suspects=local.suspects,warning='Отбор выбросов оставляет слишком мало точек; точки сохранены для ручной проверки.')
            reference = fit_arc(f[keep],z[keep],seed,robust_scale=C.INTERFERENCE_ROBUST_SCALE)
            seed = reference.params
            distance = curve_distance(f,z,reference.params)
            background = distance[~band]
            threshold = C.INTERFERENCE_MODEL_REL_FLOOR
            if len(background):
                center = float(np.median(background))
                mad = C.MAD_SCALE*float(np.median(abs(background-center)))
                threshold = max(threshold,center+C.INTERFERENCE_MODEL_MAD_MULTIPLIER*mad)
            updated = band & (distance>threshold)
            if np.array_equal(updated,excluded):
                break
            excluded = updated
        return Interference(indices[excluded],local.suspects,
                            float(np.max(distance[band])),threshold)
    except InterruptedError:
        raise
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        return Interference(suspects=local.suspects,warning=f'Не удалось надёжно определить выбросы по модели; точки сохранены: {exc}')
