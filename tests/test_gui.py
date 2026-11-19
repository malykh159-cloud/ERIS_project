import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication,QPushButton
from PySide6.QtCore import QEventLoop,QTimer,QItemSelectionModel,QCoreApplication,QEvent
from eis.parser import read_spectrum,inspect_text
from eis.window_detection import analyze
from ui.window import MainWindow, SensorWindow
from ui.dialogs import MappingDialog
from ui.plotting import export_plots
from eis.types import Result,Fit
from eis.types import Spectrum
from ui.plotting import NyquistPlot
import pyqtgraph as pg
from eis.exporter import export_summary,summary_row


@pytest.fixture(scope='session')
def app():
    application = QApplication.instance() or QApplication([])
    # Keep the application alive through pyqtgraph's atexit cleanup as well.
    pg.mkQApp()
    yield application
    # Drain offscreen widgets/deferred deletes while QApplication is still alive.
    # Otherwise native Qt objects can outlive it and crash at interpreter shutdown.
    import gc
    pg.cleanup()
    application.closeAllWindows()
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()


@pytest.fixture(autouse=True)
def drain_gui_objects(app):
    yield
    # Collect Qt reference cycles on the GUI thread, before the next worker starts.
    import gc
    for widget in app.topLevelWidgets():
        if isinstance(widget, MainWindow) and widget.plot.plotItem is not None:
            widget.plot.close()
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_point_hover_includes_excluded_frequencies(app):
    plot = NyquistPlot()
    plot.resize(800, 600)
    plot.show()
    s = Spectrum(Path('hover.txt'), np.array([200000., 100000., 1000., 10.]),
                 np.array([1-1j, 2-2j, 3-3j, 4+1j]), np.ones(4, bool))
    plot.show_result(Result(s.path, spectrum=s))
    groups = [item for item in plot.plotItem.items if isinstance(item, pg.ScatterPlotItem)]
    assert sorted(float(p.data()) for g in groups for p in g.points()) == [10., 1000., 100000., 200000.]
    assert all(g.opts['hoverable'] for g in groups)
    for group in groups:
        for point in group.points():
            tip = group.opts['tip'](point.pos().x(), point.pos().y(), point.data())
            assert f'{float(point.data()):.8g} Гц' in tip
    assert 'Выше 100 кГц (исключены)' in [g.opts['name'] for g in groups]
    app.processEvents()
    plot.show_result(None)
    assert plot.plotItem.vb.toolTip() == ''
    plot.close()


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


def test_cancel_while_waiting_for_mapping_keeps_pending_row(app,tmp_path,monkeypatch):
    path = tmp_path/'unknown_columns.txt'
    path.write_text('1000 12 -1\n100 20 -3\n10 30 -2',encoding='utf-8')
    window = MainWindow()
    window.show()
    window.add_paths([path])
    monkeypatch.setattr(window,'map_columns',lambda table: window.cancel())
    window.process_all()
    wait_worker(window)
    assert window.results[0].status == 'Не обработан'
    assert not window.results[0].reasons
    assert 'отменена' in window.statusBar().currentMessage()
    window.close()


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


def test_selected_export_uses_selected_rows_for_tables_and_png(app,tmp_path,monkeypatch):
    import csv
    from openpyxl import load_workbook
    from PySide6.QtWidgets import QFileDialog, QMessageBox, QDialog
    from ui.dialogs import ExportDialog
    from eis.fitter import fit_arc

    source = Path(__file__).resolve().parents[1]/'examples/01_clean_arc.txt'
    spectrum = read_spectrum(source)
    used = spectrum.eligible
    fit = fit_arc(spectrum.frequency[used], spectrum.z[used])
    window = MainWindow()
    window.show()
    for row in range(3):
        r = Result(tmp_path/f'measurement_{row}.txt', spectrum=spectrum, used=used, fit=fit, status='OK')
        window.results.append(r)
        window.table.insertRow(row)
        window.update_row(row,r)
    window.table.setCurrentCell(1,0)
    window.table.clearSelection()
    mode = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    for row in (0,2):
        window.table.selectionModel().select(window.table.model().index(row,0),mode)

    def accept_export(dialog):
        dialog.name.setText('Summary.v1')
        for box in dialog.formats.values():
            box.setChecked(True)
        dialog.plots.setChecked(True)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ExportDialog,'exec',accept_export)
    monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *args: str(tmp_path))
    monkeypatch.setattr(QMessageBox,'information',lambda *args: None)
    monkeypatch.setattr(QMessageBox,'warning',lambda *args: pytest.fail(str(args)))
    button = next(b for b in window.findChildren(QPushButton) if b.text()=='Экспортировать выбранные')
    button.click()
    expected = ['measurement_0.txt','measurement_2.txt']
    for fmt, delimiter in [('csv',','),('txt','\t')]:
        with (tmp_path/f'Summary.v1.{fmt}').open(encoding='utf-8-sig',newline='') as stream:
            assert [r[0] for r in list(csv.reader(stream,delimiter=delimiter))[1:]] == expected
    book = load_workbook(tmp_path/'Summary.v1.xlsx',read_only=True)
    assert [r[0] for r in book.active.iter_rows(min_row=2,values_only=True)] == expected
    book.close()
    assert len(list((tmp_path/'plots').glob('*.png'))) == 2
    assert not any('measurement_1' in p.name for p in (tmp_path/'plots').glob('*.png'))
    # Rejecting replacement must preserve the existing selected export.
    existing = (tmp_path/'Summary.v1.csv').read_bytes()
    monkeypatch.setattr(QMessageBox,'question',lambda *args: QMessageBox.StandardButton.No)
    window.export()
    assert (tmp_path/'Summary.v1.csv').read_bytes() == existing
    # General export still includes all three rows.
    monkeypatch.setattr(QMessageBox,'question',lambda *args: QMessageBox.StandardButton.Yes)
    window.export()
    with (tmp_path/'Summary.v1.csv').open(encoding='utf-8-sig',newline='') as stream:
        assert len(list(csv.reader(stream))) == 4
    window.table.clearSelection()
    monkeypatch.setattr(ExportDialog,'exec',lambda *args: pytest.fail('Empty selection opened export'))
    button.click()
    assert 'Выделите' in window.statusBar().currentMessage()
    window.close()


def test_plot_resets_after_extreme_extrapolation_and_handles_unrepresentable_peak(app):
    s = Spectrum(Path('ranges.txt'), np.geomspace(1e5,1,20),
                 np.linspace(12,100,20)-1j*np.linspace(1,50,20), np.ones(20,bool))
    plot = NyquistPlot()
    plot.resize(900,650)
    plot.show()
    for rp,q,n in [(1e12,1e-4,.5), (240.,2e-4,.85), (240.,1e-20,1e-6)]:
        fit = Fit(np.array([12.,rp,q,n]),None,0.,np.zeros(20,complex),1.,False)
        plot.show_result(Result(s.path,spectrum=s,used=s.eligible,fit=fit))
        app.processEvents()
        if rp == 240:
            assert max(abs(v) for axis in plot.viewRange() for v in axis) < 10000
        assert np.all(np.isfinite(plot.viewRange()))
    plot.reset_view()
    plot.close()

def test_sensor_selector_preserves_independent_file_lists(app):
    window = SensorWindow()
    try:
        dd = window.workspaces['DD-alphasense']
        sangbay = window.workspaces['Sangbay']
        path = Path('examples/01_clean_arc.txt')
        dd.add_paths([path])
        window.selector.setCurrentText('Sangbay')
        assert window.pages.currentWidget() is sangbay
        assert not sangbay.results
        sangbay.add_paths([path])
        assert sangbay.results[0].sangbay
        assert sangbay.boundary_cpe
        window.selector.setCurrentText('DD-alphasense')
        assert window.pages.currentWidget() is dd
        assert len(dd.results) == 1 and not dd.results[0].sangbay
    finally:
        window.close()


def test_sangbay_schemes_share_files_and_restore_model_results(app):
    window = MainWindow(sangbay=True)
    try:
        assert window.model_selector.currentIndex() == 1
        window.model_selector.setCurrentIndex(0)
        window.add_paths([Path('examples/01_clean_arc.txt')])
        original = window.results[0]
        original.notes.append('Результат первой схемы')
        window.model_selector.setCurrentIndex(1)
        assert window.results[0].path == original.path
        assert window.results[0].boundary_cpe
        assert window.table.horizontalHeaderItem(7).text().startswith('QT')
        assert window.table.columnCount() == 10
        window.model_selector.setCurrentIndex(0)
        assert window.results[0] is original
        assert window.table.horizontalHeaderItem(5).text() == 'B, √с'
        assert window.table.columnCount() == 8
        window.clear_files()
        window.model_selector.setCurrentIndex(1)
        assert not window.results
    finally:
        window.close()
