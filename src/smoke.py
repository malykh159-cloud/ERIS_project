"""End-to-end self-check available in both source and frozen executable."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PySide6.QtCore import QEventLoop,QTimer,Qt
from PySide6.QtGui import QImage
from eis.model import impedance
from eis.exporter import export_summary


def run_smoke(app,window,destination):
    from ui.plotting import export_plots
    destination = Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    f = np.logspace(5,-2,70)
    rng = np.random.default_rng(5)
    z = impedance(f,12,240,2e-4,.85)
    z += .001*abs(z)*(rng.normal(size=len(f))+1j*rng.normal(size=len(f)))
    tail = z.copy()
    tail[46:] += 80*(np.sqrt(f[46]/f[46:])-.8)*(1-1j)
    paths = []
    for name,values in [('clean',z),('tail',tail)]:
        extension, separator = ('csv', ',') if name == 'clean' else ('txt', '\t')
        path = destination/f'synthetic_{name}.{extension}'
        path.write_text('SYNTHETIC. Rs=12; Rp=240.\n' + separator.join(['Frequency','Zre','Zim']) + '\n' +
                        '\n'.join(separator.join([f'{a:.17g}', f'{b.real:.17g}', f'{b.imag:.17g}'])
                                  for a,b in zip(f,values)), encoding='utf-8')
        paths.append(path)
    bad = destination/'synthetic_invalid.txt'
    bad.write_text('not a numeric table',encoding='utf-8')
    paths.append(bad)
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    window.add_paths(paths)

    def wait():
        loop = QEventLoop()
        window.worker.finished.connect(loop.quit)
        QTimer.singleShot(60000,loop.quit)
        loop.exec()
        app.processEvents()
        assert window.worker is None, 'Worker did not complete'

    window.process_all()
    wait()
    clean,tail_result,bad_result = window.results
    assert clean.status == 'OK', clean.reasons
    assert abs(clean.fit.rs/12-1)<.02
    assert abs(clean.fit.rp/240-1)<.02
    assert 42 <= len(tail_result.used) <= 48
    assert bad_result.status == 'Ошибка импорта'
    window.table.setCurrentCell(0,0)
    window.manual_range()
    # Use actual viewport mouse events to exercise nearest-point hit testing.
    from PySide6.QtTest import QTest
    import pyqtgraph as pg
    for index in (20,44):
        point = window.plot.plotItem.vb.mapViewToScene(pg.Point(z.real[index],-z.imag[index]))
        QTest.mouseClick(window.plot.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,
                         window.plot.mapFromScene(point))
        app.processEvents()
    wait()
    assert window.results[0].manual
    assert len(window.results[0].used)==25, window.results[0].used.tolist()
    window.reset_auto()
    wait()
    export_summary(window.results,destination/'Summary',['xlsx','csv','txt'])
    export_plots(window.results,destination/'plots')
    assert len(list((destination/'plots').glob('*.png')))==2
    for path in (destination/'plots').glob('*.png'):
        image = QImage(str(path))
        assert not image.isNull() and image.width()==1600
    from openpyxl import load_workbook
    wb = load_workbook(destination/'Summary.xlsx')
    assert wb.sheetnames==['Summary'] and wb.active.max_row==4
    wb.close()
    # Exercise the selected-export button in source AND frozen builds. Capture the
    # records before the save dialogs, then verify their real table/PNG exporters.
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtWidgets import QPushButton
    import csv
    window.table.setCurrentCell(1,0)
    window.table.clearSelection()
    mode = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    for row in (0,2):
        window.table.selectionModel().select(window.table.model().index(row,0),mode)
    selected = []
    original_export = window.export_results
    try:
        window.export_results = lambda results,title: selected.extend(results)
        button = next(b for b in window.findChildren(QPushButton)
                      if b.text() == 'Экспортировать выбранные')
        button.click()
        assert selected == [window.results[0],window.results[2]]
        window.table.clearSelection()
        button.click()
        assert len(selected) == 2
    finally:
        window.export_results = original_export
    selected_dir = destination/'selected'
    selected_dir.mkdir(exist_ok=True)
    export_summary(selected,selected_dir/'Summary.v1',['xlsx','csv','txt'])
    export_plots(selected,selected_dir/'plots')
    assert len(list((selected_dir/'plots').glob('*.png'))) == 1
    with (selected_dir/'Summary.v1.csv').open(encoding='utf-8-sig',newline='') as stream:
        assert [r[0] for r in list(csv.reader(stream))[1:]] == [paths[0].name,paths[2].name]
    (destination/'selected_export_verification.json').write_text(
        json.dumps({'selected_rows':[0,2], 'current_row_not_exported':1,
                    'table_records':2, 'png_count':1, 'dotted_report_name_preserved':True},indent=2),
        encoding='utf-8')
    unmapped = destination/'synthetic_unmapped.txt'
    unmapped.write_text('1000 12 -1\n100 20 -3\n10 30 -2',encoding='utf-8')
    paths.append(unmapped)
    hashes[str(unmapped)] = hashlib.sha256(unmapped.read_bytes()).hexdigest()
    pending_row = len(window.results)
    window.add_paths([unmapped])
    original_mapping = window.map_columns
    try:
        window.map_columns = lambda table: window.cancel()
        window.process_rows([pending_row])
        wait()
        assert window.results[pending_row].status == 'Не обработан'
        assert 'отменена' in window.statusBar().currentMessage()
    finally:
        window.map_columns = original_mapping
    window.table.clearSelection()
    window.table.selectRow(pending_row)
    window.remove_selected()
    (destination/'cancellation_verified.json').write_text(
        json.dumps({'cancel_while_mapping':True,'pending_row_preserved':True,
                    'completed_results_preserved':len(window.results)==3}),encoding='utf-8')
    assert hashes == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    app.processEvents()
    window.table.setCurrentCell(1,0)
    app.processEvents()
    assert window.grab().save(str(destination/'ui_preview.png'))
    report = [{'filename':r.path.name,'status':r.status,'rs':r.fit.rs if r.fit else None,
               'rp':r.fit.rp if r.fit else None,'points_used':len(r.used),'reasons':r.reasons}
              for r in window.results]
    (destination/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'failure.txt').unlink(missing_ok=True)
