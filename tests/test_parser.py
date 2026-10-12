import numpy as np
import pytest
from eis.parser import read_spectrum, inspect_text, spectrum_from_table, Mapping


@pytest.mark.parametrize('sep,decimal,sign', [('\t','.',False), (';',',',False), (';',',',True), (',','.',False), (' ','.',True)])
@pytest.mark.parametrize('extension', ['txt', 'csv'])
def test_formats(tmp_path, sep, decimal, sign, extension):
    p = tmp_path / f'данные.{extension}'
    headers = ['Frequency', "Z'", "-Z''" if sign else "Z''"]
    rows = ['служебная строка', sep.join(headers)]
    for f, re, im in [(100,12,-1),(10,40,-20),(1,80,-10)]:
        cells = [str(float(x)).replace('.',decimal) for x in (f,re,-im if sign else im)]
        rows.append(sep.join(cells))
    p.write_text('\n'.join(rows), encoding='utf-8-sig')
    s = read_spectrum(p)
    assert s.frequency.tolist() == [100,10,1]
    assert s.z.imag.tolist() == [-1,-20,-10]


def test_manual_no_guess(tmp_path):
    p = tmp_path / 'unnamed.txt'
    p.write_text('100 12 -1\n10 20 -3\n1 40 -2')
    t = inspect_text(p)
    assert t.mapping is None
    with pytest.raises(ValueError):
        spectrum_from_table(t)
    s = spectrum_from_table(t, Mapping(0,1,2))
    assert len(s.z) == 3


def test_invalid_duplicate_and_sign(tmp_path):
    p = tmp_path / 'bad.txt'
    p.write_text("f;Re(Z);-Im(Z)\n100;12;-1\n10;20;2\n10;20;2\n0;1;2\n1;nan;5\n0.1;50;3")
    s = read_spectrum(p)
    assert s.frequency.tolist() == [100,10,0.1]
    assert s.eligible.tolist() == [1,2]


@pytest.mark.parametrize('encoding',['cp1251','cp1252','utf-8','utf-8-sig'])
def test_encodings(tmp_path, encoding):
    p = tmp_path / 'encoded.txt'
    preamble = 'измерение' if encoding == 'cp1251' else 'measurement °C'
    p.write_text(preamble + "\nf\tZre\tZim\n100\t12\t-1\n10\t20\t-3", encoding=encoding)
    assert len(read_spectrum(p).z) == 2
