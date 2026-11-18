import csv
import numpy as np
import pytest
from openpyxl import load_workbook
from eis.types import Result,Fit
from eis.exporter import export_summary,HEADERS


def test_dotted_report_name_is_preserved(tmp_path):
    result = Result(tmp_path/'measurement.txt', status='Не обработан')
    paths = export_summary([result], tmp_path/'Summary.v1', ['xlsx', 'csv', 'txt'])
    assert [p.name for p in paths] == ['Summary.v1.xlsx', 'Summary.v1.csv', 'Summary.v1.txt']
    assert not (tmp_path/'Summary.xlsx').exists()


def test_export_all_formats_and_precision(tmp_path):
    f = Fit(np.array([12.123456789,240.987654321,1e-4,.85]), np.array([.12,1.23,1e-5,.01]),
            1.23456789e-7,np.zeros(12,complex),10,False)
    f.covariance = np.diag(f.se**2)
    ok = Result(tmp_path/'=sample.txt',fit=f,used=np.arange(12),status='OK')
    bad = Result(tmp_path/'bad.txt',status='Ошибка импорта')
    paths = export_summary([ok,bad],tmp_path/'Summary',['xlsx','csv','txt'])
    assert len(paths) == 3
    wb = load_workbook(paths[0])
    assert wb.sheetnames == ['Summary']
    ws = wb.active
    assert [c.value for c in ws[1]] == HEADERS
    assert ws['A2'].data_type == 's'
    assert ws['B2'].value == f.rs
    assert ws.max_column==16
    assert ws['F2'].value==f.q
    assert ws['G2'].value==f.se[2]
    assert ws['H2'].value==f.n
    assert ws['I2'].value==f.se[3]
    assert ws['J2'].value==pytest.approx(f.c_eff)
    assert ws['K2'].value==pytest.approx(f.c_eff_se)
    assert ws['L2'].value==pytest.approx(f.f_max)
    assert ws['M2'].value==pytest.approx(f.f_max_se)
    assert ws['N2'].value==f.chi2
    assert ws['B3'].value == 'N/A'
    assert ws.max_row == 3
    wb.close()
    for path,sep in [(paths[1],','),(paths[2],'\t')]:
        with path.open(encoding='utf-8-sig',newline='') as stream:
            rows = list(csv.reader(stream,delimiter=sep))
        assert rows[0] == HEADERS
        assert float(rows[1][1]) == f.rs
        assert [float(v) for v in rows[1][5:9]]==[f.q,float(f.se[2]),f.n,float(f.se[3])]
        assert rows[2][-1] == 'Ошибка импорта'
