"""Sequential CNLS windows plus prospective residuals against the preceding fit."""
import numpy as np
from .fitter import fit_arc
from .model import impedance
from .types import Result
from .quality import assess
from .interference import detect_interference
from .geometry import tail_candidate
from . import config as C


def analyze(spectrum, manual=None, cancelled=lambda: False):
    result = Result(spectrum.path, spectrum=spectrum, manual=manual is not None)
    interference = detect_interference(spectrum,manual,cancelled)
    result.noise_indices = interference.excluded
    result.noise_suspects = interference.suspects
    result.noise_threshold = interference.threshold if len(interference.excluded) else None
    eligible = np.setdiff1d(spectrum.eligible,interference.excluded)
    if len(interference.excluded):
        result.reasons.append(f"Возможные сетевые наводки {C.INTERFERENCE_HIGH_HZ:g}–{C.INTERFERENCE_LOW_HZ:g} Гц: исключено {len(interference.excluded)} отдельных точек, отклоняющихся от модельной линии более чем на {interference.threshold:.1%} относительно |Z|. Остальные точки диапазона сохранены.")
    elif len(interference.suspects):
        result.reasons.append('Локальный разброс 300–10 Гц отмечен; существенные выбросы относительно опорной линии не подтверждены. Точки не исключены как наводки.')
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
        if manual is not None:
            best = fit_arc(f, z)
            result.used = eligible
            step = C.ENDPOINT_STABILITY_STEP
            shorter = fit_arc(f[:-step], z[:-step], best.params) if len(f) >= C.MIN_POINTS + step else best
            drift = abs(shorter.rp / best.rp - 1)
        else:
            fits, errors, history, bad_start, streak = {}, [], [], None, 0
            last_good = len(f)
            boundary_found = False
            seed = None
            crossed_guard = bool(np.any(f <= C.AUTO_BOUNDARY_MAX_HZ))
            # Use quiet anchors to check the shape when interference was observed.
            # Good band points STILL participate in every candidate CNLS and final
            # fit; only slope-based boundary corroboration ignores their jitter.
            shape = eligible
            if len(interference.suspects):
                shape = eligible[(spectrum.frequency[eligible]>C.INTERFERENCE_HIGH_HZ)|
                                 (spectrum.frequency[eligible]<C.INTERFERENCE_LOW_HZ)]
            shape_end = tail_candidate(spectrum.frequency[shape],spectrum.z[shape])
            geometric_end = int(np.searchsorted(eligible,shape[shape_end-1],side='right')) if shape_end is not None else None
            # Every endpoint is evaluated; prefix length is never selected using geometry alone.
            for k in range(C.MIN_POINTS, len(f) + 1):
                if cancelled():
                    raise InterruptedError("Обработка отменена.")
                fit = fit_arc(f[:k], z[:k], seed)
                fits[k] = fit
                seed = fit.params
                diagnostic = {"points": k, "chi2": fit.chi2, "rp": fit.rp,
                              "rp_se": None if fit.se is None else float(fit.se[1]),
                              "condition": fit.condition, "boundary": fit.boundary,
                              "end_frequency_hz": float(f[k - 1]),
                              "boundary_search_enabled": bool(f[k - 1] <= C.AUTO_BOUNDARY_MAX_HZ),
                              "tail_candidate_points": geometric_end}
                result.candidates.append(diagnostic)
                if k > C.MIN_POINTS:
                    previous = fits[k - 1]
                    prospective = abs((z[k - 1] - impedance(f[k - 1], *previous.params)) / abs(z[k - 1]))
                    errors.append(float(prospective))
                    base = np.asarray(history[-C.HISTORY_POINTS:])
                    if len(base) >= C.MIN_HISTORY:
                        med = float(np.median(base))
                        mad = C.MAD_SCALE * float(np.median(abs(base - med)))
                        limit = max(med * C.DETECTION_RATIO, med + C.DETECTION_SIGMAS * mad,
                                    C.NOISE_FLOOR)
                        # Persistent NEW point disagreement, together with growth in reduced loss.
                        log_history = [np.log10(max(fits[t].chi2, C.NOISE_FLOOR**2))
                                       for t in range(max(C.MIN_POINTS, k - C.HISTORY_POINTS), k)]
                        log_base = float(np.median(log_history))
                        log_mad = C.MAD_SCALE * float(np.median(abs(np.array(log_history) - log_base)))
                        chi_growth = np.log10(max(fit.chi2, C.NOISE_FLOOR**2)) > log_base + max(np.log10(C.CHI_GROWTH_RATIO), C.CHI_GROWTH_MAD_MULTIPLIER * log_mad)
                        # Prefix ending BEFORE the newly added point must itself have
                        # crossed 2 kHz. Reset the streak above this guard so earlier
                        # high-frequency distortions cannot later trigger a cutoff.
                        # A shape transition must corroborate model disagreement:
                        # an imperfect single-CPE fit on the rising arc alone must
                        # not cut the spectrum before the actual tail transition.
                        endpoint_allowed = (f[k - 2] <= C.AUTO_BOUNDARY_MAX_HZ and
                                            geometric_end is not None and k-1 >= geometric_end)
                        bad = endpoint_allowed and prospective > limit and (chi_growth or streak > 0)
                        if bad:
                            if streak == 0:
                                bad_start = k - 1
                            streak += 1
                        else:
                            streak, bad_start = 0, None
                            history.append(float(prospective))
                        if streak >= C.CONFIRM_POINTS:
                            last_good = min(bad_start,geometric_end+C.ENDPOINT_STABILITY_STEP)
                            boundary_found = True
                            break
                    else:
                        history.append(float(prospective))
                if geometric_end is not None and k >= geometric_end+C.CONFIRM_POINTS:
                    if not boundary_found:
                        last_good = geometric_end
                        boundary_found = True
                        result.reasons.append("Перегиб к низкочастотному хвосту обнаружен, но граница по CNLS неоднозначна; проверьте диапазон вручную.")
                    break
            best = fits[last_good]
            result.used = eligible[:last_good]
            neighbor = fits.get(last_good - C.ENDPOINT_STABILITY_STEP, best)
            drift = abs(neighbor.rp / best.rp - 1)
            if boundary_found:
                result.reasons.append(f"Низкочастотные точки исключены после устойчивого ухудшения CNLS; конец дуги {f[last_good - 1]:.6g} Гц (не выше {C.AUTO_BOUNDARY_MAX_HZ:g} Гц).")
            elif streak or (best.chi2 > C.MAX_CHI2):
                result.reasons.append("Граница дуги неоднозначна; проверьте диапазон вручную.")
        result.fit = best
        warnings = assess(best, spectrum.frequency[result.used], spectrum.z[result.used], drift)
        if interference.warning:
            warnings.append('Отбор сетевых выбросов не подтверждён моделью; проверьте точки вручную.')
        if len(interference.excluded):
            warnings.append("Часть спектра исключена из-за возможных сетевых наводок; проверьте оставшийся диапазон.")
        if manual is None and not crossed_guard:
            warnings.append(f"Спектр не достигает {C.AUTO_BOUNDARY_MAX_HZ:g} Гц; автоматический конец дуги не определён.")
        if manual is None and any("неоднозначна" in s for s in result.reasons):
            warnings.append("Автоматическая граница не подтверждена.")
        # Excluded tail is not asserted to be physically Warburg: model mismatch can be a second arc.
        if manual is None and len(result.used) < len(eligible):
            warnings.append("Автоматическая граница предполагает одну дугу; проверьте исключённый участок.")
        result.reasons.extend(warnings)
        result.status = "Проверьте диапазон" if warnings else "OK"
    except InterruptedError:
        raise
    except Exception as exc:
        result.status = "Ошибка фитирования"
        result.reasons.append(str(exc))
    return result
