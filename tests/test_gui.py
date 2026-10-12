import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication,QPushButton
from PySide6.QtCore import QEventLoop,QTimer,QItemSelectionModel
from eis.parser import read_spectrum,inspect_text
from eis.window_detection import analyze
from ui.window import MainWindow
from ui.dialogs import MappingDialog
from ui.plotting import export_plots
from eis.types import Result,Fit
from eis.exporter import export_summary,summary_row


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def wait_worker(window):
    loop = QEventLoop()
    window.worker.finished.connect(loop.quit)
    QTimer.singleShot(60000,loop.quit)
    loop.exec()
    QApplication.processEvents()
    assert window.worker is None


def test_batch_ui_manual_and_png(app,tmp_path):
    window = MainWindow()
    window.show()
    base = Path(__file__).resolve().parents[1]/'examples'
    csv_path = tmp_path/'clean.csv'
    csv_path.write_bytes((base/'01_clean_arc.txt').read_bytes())
    window.add_paths([csv_path,base/'02_diffusion_tail.txt'])
    bad = tmp_path/'bad.txt'
    bad.write_text('no numeric data')
    window.add_paths([bad])
    window.process_all()
    wait_worker(window)
    assert window.results[0].status == 'OK'
    assert len(window.results[1].used) < len(window.results[1].spectrum.z)
    assert window.results[2].status == 'Ошибка импорта'
    window.table.setCurrentCell(0,0)
    window.manual_range()
    window.range_point(5)
    window.range_point(45)
    wait_worker(window)
    assert list(window.results[0].used) == list(range(5,46))
    assert window.results[0].manual
    export_plots(window.results,tmp_path/'plots')
    assert len(list((tmp_path/'plots').glob('*.png'))) == 2
    window.reset_auto()
    wait_worker(window)
    assert not window.results[0].manual
    window.close()


def test_mapping_preview(app,tmp_path):
    p = tmp_path/'columns.txt'
    p.write_text('100 12 -1\n10 20 -3\n1 40 -2')
    dialog = MappingDialog(inspect_text(p))
    assert dialog.preview.rowCount() == 3
    dialog.validate()
    assert dialog.mapping.imag == 2
    dialog.close()


def test_process_only_selected_and_preserve_manual_range(app,tmp_path):
    window = MainWindow()
    window.show()
    base = Path(__file__).resolve().parents[1]/'examples'
    bad = tmp_path/'bad.txt'
    bad.write_text('no numeric data')
    window.add_paths([base/'01_clean_arc.txt',base/'02_diffusion_tail.txt',bad])
    window.table.clearSelection()
    window.process_selected()
    assert window.worker is None
    # Select two disjoint rows while leaving the CURRENT row unselected.
    window.table.setCurrentCell(1,0)
    window.table.clearSelection()
    select = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    for row in (0,2):
        window.table.selectionModel().select(window.table.model().index(row,0),select)
    button = next(b for b in window.findChildren(QPushButton) if b.text()=='Обработать выбранные')
    untouched = window.results[1]
    button.click()
    wait_worker(window)
    assert window.results[0].fit is not None
    assert window.results[1] is untouched and window.results[1].status=='Не обработан'
    assert window.results[2].status=='Ошибка импорта'
    # One selected row processes normally, and its manual interval survives a rerun.
    window.table.clearSelection()
    window.table.selectRow(1)
    button.click()
    wait_worker(window)
    assert window.results[1].fit is not None
    window.manual_range()
    window.range_point(8)
    window.range_point(40)
    wait_worker(window)
    expected = window.results[1].used.copy()
    others = [window.results[0],window.results[2]]
    button.click()
    wait_worker(window)
    assert window.results[1].manual
    assert np.array_equal(window.results[1].used,expected)
    assert window.results[0] is others[0] and window.results[2] is others[1]
    window.close()


def test_table_precision_does_not_round_export(app,tmp_path):
    import csv
    from openpyxl import load_workbook
    fit = Fit(np.array([12.123456789,240.987654321,1e-4,.85]),
              np.array([.0638862,1.234567,1e-5,.01]),1.23456789e-7,np.zeros(12,complex),10,False)
    result = Result(tmp_path/'precision.txt',fit=fit,used=np.arange(12),status='OK')
    before = summary_row(result)
    window = MainWindow()
    window.results.append(result)
    window.table.insertRow(0)
    window.update_row(0,result)
    assert [window.table.item(0,i).text() for i in range(1,5)]==['12,12','241,0','1,00e-04','0,85']
    assert window.table.item(0,7).text()=='1,2e-07'
    assert all('SE' not in window.table.horizontalHeaderItem(i).text() for i in range(window.table.columnCount()))
    assert summary_row(result)==before
    paths = export_summary(window.results,tmp_path/'Summary',['xlsx','csv','txt'])
    wb = load_workbook(paths[0],read_only=True)
    values = list(wb.active.iter_rows(min_row=2,values_only=True))[0]
    for actual,expected in zip(values[1:-2],before[1:-2]):
        assert actual=='N/A' if expected is None else actual==pytest.approx(expected,rel=1e-14)
    wb.close()
    for path,sep in [(paths[1],','),(paths[2],'\t')]:
        with path.open(encoding='utf-8-sig',newline='') as stream:
            rows = list(csv.reader(stream,delimiter=sep))
        for actual,expected in zip(rows[1][1:-2],before[1:-2]):
            assert actual=='N/A' if expected is None else float(actual)==expected
    window.close()
