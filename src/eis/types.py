from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property
import numpy as np
from .derived import arc_characteristics
from .model import impedance
from . import sangbay_model
from . import config as C


@dataclass
class Spectrum:
    path: Path
    frequency: np.ndarray
    z: np.ndarray
    valid: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def eligible(self):
        return np.flatnonzero(self.valid & (self.z.imag <= 0) &
                              (self.frequency <= C.MAX_FIT_FREQUENCY_HZ))


@dataclass
class Fit:
    params: np.ndarray
    se: np.ndarray | None
    chi2: float
    residual: np.ndarray
    condition: float
    boundary: bool
    stable: bool = True
    covariance: np.ndarray | None = None

    def predict(self, frequency):
        return impedance(frequency, *self.params)

    @property
    def rms_percent(self):
        return float(100 * np.sqrt(np.mean(np.abs(self.residual)**2)))

    @property
    def rs(self):
        return float(self.params[0])

    @property
    def rp(self):
        return float(self.params[1])

    @property
    def q(self):
        return float(self.params[2])

    @property
    def n(self):
        return float(self.params[3])

    @property
    def c_eff(self):
        return arc_characteristics(self.params,self.covariance)[0]

    @property
    def f_max(self):
        return arc_characteristics(self.params,self.covariance)[1]

    @property
    def c_eff_se(self):
        return arc_characteristics(self.params,self.covariance)[2]

    @property
    def f_max_se(self):
        return arc_characteristics(self.params,self.covariance)[3]


@dataclass
class SangbayFit(Fit):
    frequency_bounds: tuple[float, float] = (0.0, 0.0)

    def predict(self, frequency):
        return sangbay_model.impedance(frequency, *self.params)

    @property
    def b(self):
        return float(self.params[4])

    @property
    def rd(self):
        return float(self.params[5])

    @cached_property
    def characteristics(self):
        c_eff, peak = sangbay_model.peak_characteristics(self.params, *self.frequency_bounds)
        if peak is None or self.covariance is None or not self.stable:
            return c_eff, peak, None, None
        gradients = []
        for index, value in enumerate(self.params):
            if index == 0:  # Rs shifts only the real impedance.
                gradients.append([0.0, 0.0])
                continue
            step = value * 1e-3
            low, high = self.params.copy(), self.params.copy()
            low[index] -= step
            high[index] += step
            a = sangbay_model.peak_characteristics(low, *self.frequency_bounds)
            b = sangbay_model.peak_characteristics(high, *self.frequency_bounds)
            if a[1] is None or b[1] is None:
                return c_eff, peak, None, None
            gradients.append((np.asarray(b) - a) / (2 * step))
        g = np.asarray(gradients).T
        variances = np.diag(g @ self.covariance @ g.T)
        errors = np.sqrt(np.maximum(variances, 0))
        return c_eff, peak, *(float(e) if np.isfinite(e) else None for e in errors)

    c_eff = property(lambda self: self.characteristics[0])
    f_max = property(lambda self: self.characteristics[1])
    c_eff_se = property(lambda self: self.characteristics[2])
    f_max_se = property(lambda self: self.characteristics[3])


class BoundaryFit(Fit):
    def predict(self, frequency):
        return sangbay_model.boundary_impedance(frequency, *self.params)


@dataclass
class Result:
    path: Path
    spectrum: Spectrum | None = None
    used: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    fit: Fit | None = None
    status: str = "Не обработан"
    reasons: list[str] = field(default_factory=list)
    manual: bool = False
    candidates: list[dict] = field(default_factory=list)
    noise_indices: np.ndarray = field(default_factory=lambda: np.array([],dtype=int))
    noise_suspects: np.ndarray = field(default_factory=lambda: np.array([],dtype=int))
    noise_threshold: float | None = None
    notes: list[str] = field(default_factory=list)
    unstable_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    sangbay: bool = False
    boundary_cpe: bool = False
