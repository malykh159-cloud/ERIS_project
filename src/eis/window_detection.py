"""Select the one-arc fitting window and document low-frequency exclusions."""
import numpy as np
from scipy.signal import find_peaks
from .fitter import fit_arc
from .model import impedance
from .types import Result
from .quality import assess, requires_far_extrapolation
from .interference import detect_interference, local_deviation
from .geometry import tail_candidates
from . import config as C


def relative_residual(f, z, fit):
    return (z - impedance(f, *fit.params)) / np.maximum(abs(z), np.finfo(float).tiny)


def residual_threshold(values, floor, sigmas=C.DETECTION_SIGMAS):
    values = abs(values)
    center = np.median(values)
    mad = C.MAD_SCALE * np.median(abs(values - center))
    return max(floor, center + sigmas * mad)


def confirmed_tail_start(f, z, get_fit, *, allow_shoulder=False):
    checked = set()
    for candidate in tail_candidates(f, z, allow_shoulder=allow_shoulder):
        if candidate in checked:
            continue
        checked.add(candidate)
        fit = get_fit(candidate)
        threshold = residual_threshold(fit.residual[-C.HISTORY_POINTS:], C.TAIL_RESIDUAL_FLOOR)
        residual = relative_residual(f[candidate:candidate + C.TAIL_CHECK_POINTS],
                                     z[candidate:candidate + C.TAIL_CHECK_POINTS], fit)
        for offset in range(len(residual) - C.CONFIRM_POINTS + 1):
            group = residual[offset:offset + C.CONFIRM_POINTS]
            if (np.all(abs(group) > threshold)
                    and abs(np.sum(group)) >= C.TAIL_RESIDUAL_ALIGNMENT * np.sum(abs(group))):
                return candidate + offset
    return None


def gradual_tail_starts(f, z, get_fit, current_end):
    """Look back for the start of a growing deviation confirmed farther down."""
    for candidate in dict.fromkeys(tail_candidates(f, z, allow_shoulder=True)):
        if candidate >= current_end:
            continue
        fit = get_fit(candidate)
        background = fit.residual[-C.HISTORY_POINTS:]
        threshold = residual_threshold(background, C.TAIL_RESIDUAL_FLOOR)
        early_threshold = residual_threshold(background, C.TAIL_RESIDUAL_FLOOR,
                                             C.TAIL_GRADUAL_SIGMAS)
        residual = relative_residual(f[candidate:], z[candidate:], fit)
        magnitude = abs(residual)
        for offset in range(min(current_end - candidate, len(residual)) - C.CONFIRM_POINTS + 1):
            group = residual[offset:offset + C.CONFIRM_POINTS]
            levels = magnitude[offset:offset + C.CONFIRM_POINTS]
            if (np.all(levels > early_threshold) and np.all(np.diff(levels) > 0)
                    and abs(np.sum(group)) >= C.TAIL_RESIDUAL_ALIGNMENT * np.sum(levels)
                    and np.any(magnitude[offset:offset + C.TAIL_GRADUAL_CONFIRM_WINDOW] > threshold)):
                yield candidate + offset


def unstable_tail_start(f, z, get_fit):
    """Find a sustained cluster of irregular points beyond a usable arc prefix."""
    deviation, valid = local_deviation(f, z)
    scatter = np.r_[0.0, np.where(valid, deviation, 0.0), 0.0]
    width = C.UNSTABLE_TAIL_WINDOW
    for start in range(C.MIN_POINTS, len(f) - width + 1):
        if f[start - 1] > C.AUTO_BOUNDARY_MAX_HZ:
            continue
        stop = start + width
        threshold_local = residual_threshold(scatter[max(0, start - C.HISTORY_POINTS):start],
                                             C.UNSTABLE_TAIL_LOCAL_FLOOR)
        irregular = scatter[start:stop] > threshold_local
        if np.count_nonzero(irregular) < C.CONFIRM_POINTS:
            continue
        # Leave a buffer so the first irregular points cannot pull the reference
        # toward the low-frequency cluster before it is confirmed.
        fit = get_fit(max(C.MIN_POINTS, start - width))
        threshold = residual_threshold(fit.residual[-C.HISTORY_POINTS:], C.UNSTABLE_TAIL_MODEL_FLOOR)
        disagreement = abs(relative_residual(f[start:stop], z[start:stop], fit)) > threshold
        confirmed = np.flatnonzero(irregular & disagreement)
        if len(confirmed) >= C.CONFIRM_POINTS:
            quiet = scatter[stop:-1] <= threshold_local
            if len(quiet) >= width and np.any(
                    np.all(np.lib.stride_tricks.sliding_window_view(quiet, width), axis=1)):
                continue
            return start + int(confirmed[0])
    return None


def isolated_suspects(f, z, fit):
    deviation, valid = local_deviation(f, z)
    if not np.any(valid):
        return np.array([], dtype=int)
    local_limit = residual_threshold(deviation[valid], C.UNSTABLE_TAIL_LOCAL_FLOOR)
    error = abs(fit.residual)
    model_limit = residual_threshold(error, C.UNSTABLE_TAIL_MODEL_FLOOR)
    isolated = (error[1:-1] > model_limit) & (error[:-2] <= model_limit) & (error[2:] <= model_limit)
    return np.flatnonzero(valid & (deviation > local_limit) & isolated) + 1


def analyze(spectrum, manual=None, cancelled=lambda: False, *, sangbay=False, boundary_cpe=False):
    sangbay = sangbay or boundary_cpe
    result = Result(spectrum.path, spectrum=spectrum, manual=manual is not None,
                    sangbay=sangbay, boundary_cpe=boundary_cpe)
    interference = (detect_interference(spectrum, manual, cancelled, sangbay=True, boundary_cpe=boundary_cpe)
                    if sangbay else detect_interference(spectrum,manual,cancelled))
    result.noise_indices = interference.excluded
    result.noise_suspects = interference.suspects
    result.noise_threshold = interference.threshold if len(interference.excluded) else None
    eligible = np.setdiff1d(spectrum.eligible,interference.excluded)
    if len(interference.excluded):
        result.notes.append(f"Возможные сетевые наводки {C.INTERFERENCE_HIGH_HZ:g}–{C.INTERFERENCE_LOW_HZ:g} Гц: исключено {len(interference.excluded)} отдельных точек, отклоняющихся от модельной линии более чем на {interference.threshold:.1%} относительно |Z|. Остальные точки диапазона сохранены.")
    elif len(interference.suspects):
        result.notes.append(f'Локальный разброс {C.INTERFERENCE_HIGH_HZ:g}–{C.INTERFERENCE_LOW_HZ:g} Гц отмечен; существенные выбросы относительно опорной линии не подтверждены. Точки не исключены как наводки.')
    if interference.warning:
        result.reasons.append(interference.warning)
    if manual is not None:
        a, b = sorted(manual)
        eligible = eligible[(eligible >= a) & (eligible <= b)]
    f, z = spectrum.frequency[eligible], spectrum.z[eligible]
    if len(f) < C.MIN_POINTS:
        result.status = "Ошибка фитирования"
        result.reasons.append(f"Осталось {len(f)} точек; требуется минимум {C.MIN_POINTS}.")
        return result
    try:
        if sangbay:
            result.fit = fit_arc(f, z, sangbay=True, boundary_cpe=boundary_cpe, cancelled=cancelled)
            result.used = eligible
            result.reasons.extend(assess(result.fit, f, z, notes=result.notes, sangbay=True))
            result.status = 'Проверьте диапазон' if result.reasons else 'OK'
            return result
        second_arc = False
        if manual is not None:
            best = fit_arc(f, z)
            result.used = eligible
            step = C.ENDPOINT_STABILITY_STEP
            shorter = fit_arc(f[:-step], z[:-step], best.params) if len(f) >= C.MIN_POINTS + step else best
            drift = abs(shorter.rp / best.rp - 1)
        else:
            crossed_guard = bool(np.any(f <= C.AUTO_BOUNDARY_MAX_HZ))
            fits = {}

            def get_fit(end):
                if cancelled():
                    raise InterruptedError("Обработка отменена.")
                if end not in fits:
                    fit = fits[end] = fit_arc(f[:end], z[:end])
                    result.candidates.append({
                        "points": end, "chi2": fit.chi2, "rp": fit.rp,
                        "end_frequency_hz": float(f[end - 1]),
                    })
                return fits[end]

            unstable = unstable_tail_start(f, z, get_fit)
            shape_end = len(f) if unstable is None else unstable
            tail = confirmed_tail_start(f[:shape_end], z[:shape_end], get_fit)
            last_good = shape_end if tail is None else tail

            def fit_and_drift(end):
                fit = get_fit(end)
                shorter = end - C.ENDPOINT_STABILITY_STEP
                drift = abs(get_fit(shorter).rp / fit.rp - 1) if shorter >= C.MIN_POINTS else 0.0
                return fit, drift

            best, drift = fit_and_drift(last_good)
            if requires_far_extrapolation(best, z[:last_good]):
                shoulder = confirmed_tail_start(f[:shape_end], z[:shape_end], get_fit, allow_shoulder=True)
                if shoulder is not None and shoulder < last_good:
                    alternative, alternative_drift = fit_and_drift(shoulder)
                    if not assess(alternative, f[:shoulder], z[:shoulder], alternative_drift):
                        tail = last_good = shoulder
                        best, drift = alternative, alternative_drift
                        result.notes.append("Граница уточнена по плечу дуги: исходный диапазон требовал далёкой экстраполяции Rp.")
            if tail is None or best.chi2 > C.MAX_CHI2:
                for end in gradual_tail_starts(f[:shape_end], z[:shape_end], get_fit, last_good):
                    alternative, alternative_drift = fit_and_drift(end)
                    if (alternative.chi2 < best.chi2 * C.TAIL_GRADUAL_MAX_CHI2_RATIO
                            and not assess(alternative, f[:end], z[:end], alternative_drift)):
                        tail = last_good = end
                        best, drift = alternative, alternative_drift
                        result.notes.append("Граница уточнена по постепенному росту отклонения от дуги.")
                        break
            result.used = eligible[:last_good]
            suspects = isolated_suspects(f[:last_good], z[:last_good], best)
            if len(suspects):
                frequencies = ', '.join(f'{value:.6g}' for value in f[suspects[:8]])
                suffix = ' и другие' if len(suspects) > 8 else ''
                result.notes.append(
                    f"Отдельные подозрительные точки оставлены в подгонке: {frequencies} Гц{suffix}."
                )
            if tail is not None:
                result.notes.append(
                    f"Низкочастотный хвост исключён по изменению наклона и согласованному "
                    f"отклонению от модели; конец дуги {f[tail - 1]:.6g} Гц."
                )
                height = -z[tail:shape_end].imag
                prominence = C.SECOND_ARC_PROMINENCE_REL * np.ptp(z[:tail].imag)
                peaks, _ = find_peaks(height, prominence=prominence, width=2)
                second_arc = bool(len(peaks))
            if unstable is not None:
                result.unstable_indices = eligible[unstable:]
                result.notes.append(
                    f"Нестабильный низкочастотный участок: {len(result.unstable_indices)} точек "
                    f"от {f[unstable]:.6g} до {f[-1]:.6g} Гц вне подгонки."
                )
            if tail is None and unstable is None and best.chi2 > C.MAX_CHI2:
                result.notes.append("Устойчивый конец дуги не найден; использован доступный диапазон.")
        result.fit = best
        warnings = assess(best, spectrum.frequency[result.used], spectrum.z[result.used], drift, result.notes)
        if second_arc:
            warnings.append("За границей подгонки есть отдельный максимум; возможна вторая дуга.")
        if interference.warning:
            warnings.append('Отбор сетевых выбросов не подтверждён моделью; проверьте точки вручную.')
        if manual is None and not crossed_guard:
            warnings.append(f"Спектр не достигает {C.AUTO_BOUNDARY_MAX_HZ:g} Гц; автоматический конец дуги не определён.")
        # Automatic exclusions remain documented; numeric quality controls acceptance.
        if manual is None and len(result.used) < len(eligible):
            result.notes.append("Обработан участок одной дуги; низкочастотная часть сохранена на графике и не входит в fit.")
        result.reasons.extend(warnings)
        result.status = "Проверьте диапазон" if warnings else "OK"
    except InterruptedError:
        raise
    except Exception as exc:
        result.status = "Ошибка фитирования"
        result.reasons.append(str(exc))
    return result
