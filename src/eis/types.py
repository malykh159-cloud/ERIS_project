from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .derived import arc_characteristics


@dataclass
class Spectrum:
    path: Path
    frequency: np.ndarray
    z: np.ndarray
    valid: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def eligible(self):
        return np.flatnonzero(self.valid & (self.z.imag <= 0))


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
