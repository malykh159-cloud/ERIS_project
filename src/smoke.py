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
