import numpy as np
from . import config as C


def assess(fit, frequency, z, drift=0.0):
    reasons = []
    if fit.c_eff is None or fit.f_max is None:
        reasons.append('Эффективная ёмкость или частота максимума не представимы численно; проверьте модель и диапазон.')
    if len(z) < C.MIN_RELIABLE_POINTS:
        reasons.append("Мало точек для надёжной экстраполяции.")
    if fit.se is None:
        reasons.append("Ковариация плохо обусловлена; стандартная ошибка недоступна.")
    elif fit.se[1] / fit.rp > C.MAX_RELATIVE_RP_SE:
        reasons.append("Rp определяется с большой относительной неопределённостью.")
    if fit.chi2 > C.MAX_CHI2:
        reasons.append("Модель одной дуги плохо описывает данные: повышенные комплексные остатки.")
    if fit.boundary:
        reasons.append("Параметр достиг границы допустимого диапазона.")
    if not fit.stable:
        reasons.append("Различные начальные приближения дают неустойчивое решение.")
    if drift > C.MAX_RP_DRIFT:
        reasons.append("Rp сильно меняется при небольшом изменении диапазона.")
    if np.log10(max(frequency) / min(frequency)) < C.MIN_LOG_SPAN:
        reasons.append("Слишком короткий частотный диапазон.")
    if fit.rp > C.MAX_EXTRAPOLATION_RATIO * max(np.ptp(z.real), np.finfo(float).tiny):
        reasons.append("Правая часть дуги почти не измерена; Rp требует далёкой экстраполяции.")
    # Detect systematic structure only above numerical/very-small-noise floor.
    for residual in (fit.residual.real, fit.residual.imag):
        if len(residual) > C.MIN_RELIABLE_POINTS and np.std(residual) > C.NOISE_FLOOR:
            corr = np.corrcoef(residual[:-1], residual[1:])[0, 1]
            if np.isfinite(corr) and corr > C.SYSTEMATIC_CORRELATION:
                reasons.append("Систематические остатки: возможны вторая дуга или индуктивное искажение.")
                break
    return reasons
