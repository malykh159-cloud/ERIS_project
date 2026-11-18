import hashlib
from pathlib import Path
import numpy as np
import pytest
from eis.parser import read_spectrum,inspect_text


FILES = sorted((Path(__file__).resolve().parents[1]/'data_test').glob('*.txt'))


@pytest.mark.parametrize('path',FILES,ids=lambda p:p.name)
def test_actual_smartstat_export(path):
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    table = inspect_text(path)
    smartstat = table.headers==['N','F, Гц','Re, Ом','Im, Ом']
    if smartstat:
        assert table.encoding=='cp1251'
    else:
        assert table.headers[:4]==['Index','Frequency (Hz)',"Z' (Ω)","-Z'' (Ω)"]
        assert table.encoding=='utf-8-sig'
    assert table.mapping.negative_imag
    spectrum = read_spectrum(path)
    assert len(spectrum.z)==len(path.read_bytes().decode(table.encoding).splitlines())-1
    first = np.array([float(v.replace(',','.')) for v in table.rows[0]])
    assert spectrum.frequency[0]==first[1]
    assert spectrum.z[0].real==first[2]
    assert spectrum.z[0].imag==-first[3]
    assert np.all(np.diff(spectrum.frequency)<=0)
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()


def test_smartstat_sign_excludes_negative_displayed_imag(tmp_path):
    path = tmp_path/'smartstat.txt'
    path.write_text('N            F, Гц            Re, Ом            Im, Ом\n'
                    '1            50000            12,5            -0,8\n'
                    '2            2000            13,5            2,8\n'
                    '3            10            22,5            12,8',encoding='cp1251')
    s = read_spectrum(path)
    assert s.eligible.tolist()==[1,2]


def test_generic_im_keeps_mathematical_sign(tmp_path):
    path = tmp_path/'generic.csv'
    path.write_text('Frequency,Re,Im\n2000,12,-2\n100,15,-4')
    assert read_spectrum(path).z.imag.tolist()==[-2,-4]
