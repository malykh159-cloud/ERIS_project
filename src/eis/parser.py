"""Extensible delimited-text importer. Ambiguous columns require explicit mapping."""
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .types import Spectrum
from . import config as C


@dataclass(frozen=True)
class Mapping:
    frequency: int
    real: int
    imag: int
    negative_imag: bool = False
    frequency_scale: float = 1.0


@dataclass
class Table:
    path: Path
    headers: list[str]
    rows: list[list[str]]
    delimiter: str
    encoding: str
    mapping: Mapping | None
    skipped: int = 0

    @property
    def signature(self):
        return (tuple(self.headers), self.delimiter, self.encoding)


def number(text):
    return float(text.strip().replace('−', '-').replace(',', '.'))


def tokens(line, delimiter):
    if delimiter == 'aligned':
        # Preserve single spaces inside unit-bearing headers (F, Гц; Re, Ом).
        return re.split(r'(?:\s{2,}|\t+)', line.strip())
    if delimiter == 'whitespace':
        return re.split(r'\s+', line.strip())
    return [s.strip() for s in next(csv.reader([line], delimiter=delimiter, quotechar='"'))]


def header_kind(text):
    s = text.lower().strip().replace('′', "'").replace('″', "''").replace('−', '-')
    frequency_unit = re.search(r'\b(khz|mhz)\b', s)
    s = re.sub(r'\[[^\]]*\]|\([^)]*(?:ohm|ω|hz|ом)[^)]*\)', '', s)
    s = re.sub(r'\s+', '', s).replace('ω', '').replace('ohms', '').replace('ohm', '').replace('ом', '')
    s = s.replace(',', '')
    # Strip units without removing the mathematical parentheses in Re(Z)/Im(Z).
    s = s.rstrip('/,;:')
    freq = re.fullmatch(r'(frequency|freq|f|hz|частота)(?:[/_\-]?(khz|mhz|hz|гц))?', s)
    if freq:
        unit = frequency_unit[1] if frequency_unit else freq[2]
        return ('frequency', {'khz': 1e3, 'mhz': 1e6}.get(unit, 1.0))
    if s in ("z'", 'zre', 're(z)', 'zreal', 'real', 're', 'действительная'):
        return ('real', False)
    negative = s.startswith('-')
    s = s.lstrip('-')
    if s in ("z''", 'zim', 'im(z)', 'zimag', 'imag', 'im', 'мнимая'):
        return ('imag', negative)
    return None


def is_smartstat_export(headers):
    # Exact supplied export profile. Do not infer sign from the numeric majority
    # or reinterpret generic Im/Zim headers in other formats.
    canonical = tuple(re.sub(r'\s+', '', h.lower()) for h in headers)
    return canonical == ('n', 'f,гц', 're,ом', 'im,ом')


def auto_mapping(headers):
    if is_smartstat_export(headers):
        # User confirmed: this file's positive Im column is Nyquist -Im(Z).
        return Mapping(1, 2, 3, negative_imag=True)
    found = {}
    for i, h in enumerate(headers):
        kind = header_kind(h)
        if kind:
            found.setdefault(kind[0], []).append((i, kind[1]))
    if any(len(found.get(k, [])) != 1 for k in ('frequency', 'real', 'imag')):
        return None
    return Mapping(found['frequency'][0][0], found['real'][0][0], found['imag'][0][0],
                   found['imag'][0][1], found['frequency'][0][1])


def inspect_text(path, delimiter=None, encoding=None):
    path = Path(path)
    raw = path.read_bytes()
    if encoding:
        text = raw.decode(encoding)
    else:
        try:
            text = raw.decode('utf-8-sig')
            encoding = 'utf-8-sig'
        except UnicodeDecodeError:
            # These single-byte encodings cannot always be distinguished; expose override in UI.
            options = []
            for enc in ('cp1251', 'cp1252'):
                try:
                    decoded = raw.decode(enc)
                    known = len(re.findall(r'частота|измерение|импеданс|данные', decoded.lower()))
                    options.append((known, enc, decoded))
                except UnicodeDecodeError:
                    pass
            if not options:
                raise ValueError('Не удалось прочитать кодировку файла.')
            _, encoding, text = max(options, key=lambda o: o[0])
    lines = text.splitlines()
    candidates = []
    for sep in ([delimiter] if delimiter else ['\t', ';', ',', 'aligned', 'whitespace']):
        split = [tokens(line, sep) for line in lines]
        numeric = []
        for i, row in enumerate(split):
            count = 0
            for cell in row:
                try:
                    number(cell)
                    count += 1
                except ValueError:
                    pass
            if len(row) >= 3 and count >= max(3, len(row) * 0.7):
                numeric.append(i)
        if not numeric:
            continue
        width = Counter(len(split[i]) for i in numeric).most_common(1)[0][0]
        numeric = [i for i in numeric if len(split[i]) == width]
        first = numeric[0]
        headers = [f'Колонка {i + 1}' for i in range(width)]
        mapping = None
        header_row = None
        for i in range(first - 1, -1, -1):
            if len(split[i]) == width:
                candidate_mapping = auto_mapping(split[i])
                if candidate_mapping:
                    headers, mapping, header_row = split[i], candidate_mapping, i
                    break
                if header_row is None and any(header_kind(h) for h in split[i]):
                    headers, header_row = split[i], i
        # A numeric metadata line can precede the real header. Recognized headers
        # take precedence over such lines and prevent metadata becoming EIS points.
        for i,row in enumerate(split):
            if len(row) == width and any(j > i for j in numeric):
                recognized = auto_mapping(row)
                if recognized is not None:
                    headers,mapping,header_row = row,recognized,i
                    break
        start = header_row + 1 if header_row is not None else first
        rows = [row for row in split[start:] if len(row) == width and any(c.strip() for c in row)]
        skipped = sum(bool(lines[i].strip()) for i in range(start, len(lines)) if len(split[i]) != width)
        score = len(numeric) + (len(lines) + 1 if mapping else 0)
        candidates.append((score, Table(path, headers, rows, sep, encoding, mapping, skipped)))
    if not candidates:
        raise ValueError('Таблица не найдена. Нужны минимум три числовых столбца; проверьте разделитель.')
    return max(candidates, key=lambda c: c[0])[1]


def spectrum_from_table(table, mapping=None):
    mapping = mapping or table.mapping
    if mapping is None:
        raise ValueError('Необходимо вручную выбрать частоту, Z′ и Z″ и соглашение о знаке.')
    indices = (mapping.frequency, mapping.real, mapping.imag)
    if len(set(indices)) != 3 or min(indices) < 0 or max(indices) >= len(table.headers):
        raise ValueError('Выберите три разных столбца таблицы.')
    rows, dropped = [], table.skipped
    for row in table.rows:
        try:
            f, re_z, im_z = [number(row[i]) for i in indices]
            f *= mapping.frequency_scale
            if mapping.negative_imag:
                im_z = -im_z
            if not np.all(np.isfinite([f, re_z, im_z])) or f <= 0:
                dropped += 1
                continue
            rows.append((f, re_z, im_z))
        except (ValueError, IndexError):
            dropped += 1
    if not rows:
        raise ValueError('После удаления некорректных строк не осталось данных.')
    data = np.array(rows)
    # Only exact duplicates are excluded. Repeated frequencies with differing Z remain measurements.
    _, keep = np.unique(data, axis=0, return_index=True)
    duplicates = len(data) - len(keep)
    data = data[np.sort(keep)]
    data = data[np.argsort(-data[:, 0], kind='stable')]
    z = data[:, 1] + 1j * data[:, 2]
    notes = [f'Кодировка: {table.encoding}; пропущено строк: {dropped}; точных дубликатов: {duplicates}.',
             f'Точек выше {C.MAX_FIT_FREQUENCY_HZ / 1000:g} кГц, исключённых из обработки: {np.sum(data[:, 0] > C.MAX_FIT_FREQUENCY_HZ)}.',
             f'Индуктивных точек ниже оси, исключённых из fit: {np.sum(z.imag > 0)}.']
    if is_smartstat_export(table.headers):
        interpretation = '−Im(Z)' if mapping.negative_imag else 'Im(Z)'
        notes.append(f'Формат SmartStat N / F, Гц / Re, Ом / Im, Ом; столбец Im прочитан как {interpretation}.')
    return Spectrum(table.path, data[:, 0], z, np.ones(len(z), dtype=bool), notes)


def read_spectrum(path, mapping=None):
    return spectrum_from_table(inspect_text(path), mapping)
