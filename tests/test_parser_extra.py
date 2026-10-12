import pytest
from eis.parser import read_spectrum,spectrum_from_table,inspect_text,Mapping


def test_numeric_metadata_before_headers(tmp_path):
    path = tmp_path/'metadata.txt'
    path.write_text("2026 10 8\nFrequency Zre Zim\n100 12 -1\n10 20 -3\n1 40 -2")
    assert read_spectrum(path).frequency.tolist()==[100,10,1]


def test_frequency_units(tmp_path):
    path = tmp_path/'units.txt'
    path.write_text("Frequency [kHz];Re(Z) [Ohm];-Im(Z) [Ohm]\n10;12;1\n1;20;3\n0.1;40;2")
    assert read_spectrum(path).frequency.tolist()==[10000,1000,100]


def test_quoted_decimal_comma(tmp_path):
    path = tmp_path/'quoted.txt'
    path.write_text('Frequency,Zre,Zim\n"100,0","12,0","-1,0"\n"10,0","20,0","-3,0"')
    assert read_spectrum(path).z.imag.tolist()==[-1,-3]


def test_mapping_columns_must_differ(tmp_path):
    path = tmp_path/'mapping.txt'
    path.write_text('100 12 -1\n10 20 -3')
    with pytest.raises(ValueError,match='разных'):
        spectrum_from_table(inspect_text(path),Mapping(0,0,2))
