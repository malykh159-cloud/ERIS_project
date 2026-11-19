"""End-to-end GUI batch validation on actual user-provided measurements."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from PySide6.QtCore import QEventLoop,QTimer
from eis import config as C
from eis.exporter import export_summary
from eis.sangbay_model import BOUNDARY_PARAMETERS, TRANSMISSIVE_PARAMETERS


def run_measurement_check(app,window,source,destination):
    from ui.plotting import export_plots
    destination = Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    paths = sorted(p.resolve() for p in Path(source).glob('*.txt'))
    if not paths:
        raise ValueError('В выбранной папке нет TXT для проверки.')
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    start = perf_counter()
    window.add_paths(paths)
    mapping_requests = []

    def unexpected_mapping(table):
        mapping_requests.append(table.path.name)
        window.worker.mapping_reply = None
        window.worker.mapping_event.set()

    def wait():
        loop = QEventLoop()
        window.worker.finished.connect(loop.quit)
        QTimer.singleShot(300000,loop.quit)
        loop.exec()
        app.processEvents()
        assert window.worker is None,'Пакетная обработка не завершилась за 5 минут.'

    # Connect before starting, so an unexpected parser regression cannot wait for a modal dialog.
    window.map_columns = unexpected_mapping
    window.process_all()
    window.worker.progress.connect(lambda n,total:(destination/'progress.json').write_text(
        json.dumps({'completed':n,'total':total}),encoding='utf-8'))
    wait()
    assert not mapping_requests, mapping_requests
    assert len(window.results)==len(paths)
    rows = []
    for r in window.results:
        assert r.spectrum is not None and r.fit is not None,(r.path.name,r.status,r.reasons)
        assert np.isfinite(r.fit.rs) and np.isfinite(r.fit.rp)
        f = r.spectrum.frequency[r.used]
        if r.sangbay:
            np.testing.assert_array_equal(r.used, np.setdiff1d(r.spectrum.eligible, r.noise_indices))
        if np.any(r.spectrum.frequency[r.spectrum.eligible]<=C.AUTO_BOUNDARY_MAX_HZ):
            assert min(f)<=C.AUTO_BOUNDARY_MAX_HZ
        assert not np.any(np.isin(r.used,r.noise_indices))
        noise_f = r.spectrum.frequency[r.noise_indices]
        assert np.all((noise_f>=C.INTERFERENCE_LOW_HZ)&(noise_f<=C.INTERFERENCE_HIGH_HZ))
        rows.append({'filename':r.path.name,'sha256':hashes[str(r.path)],'status':r.status,
                     'points_imported':len(r.spectrum.z),'points_used':len(r.used),
                     'fit_start_hz':float(max(f)),'fit_end_hz':float(min(f)),
                     'noise_excluded':len(r.noise_indices),'noise_suspects':len(r.noise_suspects),
                     'noise_frequencies_hz':r.spectrum.frequency[r.noise_indices].tolist(),
                     'rs_ohm':r.fit.rs,'rs_se_ohm':None if r.fit.se is None else float(r.fit.se[0]),
                     'rp_ohm':r.fit.rp,'rp_se_ohm':None if r.fit.se is None else float(r.fit.se[1]),
                     'q_S_s_n':r.fit.q,'q_se_S_s_n':None if r.fit.se is None else float(r.fit.se[2]),
                     'n':r.fit.n,'n_se':None if r.fit.se is None else float(r.fit.se[3]),
                     'noise_threshold_relative':r.noise_threshold,
                     'chi2':r.fit.chi2,'notes':r.spectrum.notes,'analysis_notes':r.notes,'reasons':r.reasons})
        if r.sangbay:
            parameters = BOUNDARY_PARAMETERS if r.boundary_cpe else TRANSMISSIVE_PARAMETERS
            rows[-1].update(
                relative_rms_percent=r.fit.rms_percent,
                parameters={name: float(r.fit.params[i]) for i, (_, name, _) in enumerate(parameters)},
                standard_errors={name: None if r.fit.se is None else float(r.fit.se[i])
                                 for i, (_, name, _) in enumerate(parameters)})
        else:
            rows[-1].update(c_eff_F=r.fit.c_eff, c_eff_se_F=r.fit.c_eff_se,
                            f_max_Hz=r.fit.f_max, f_max_se_Hz=r.fit.f_max_se)
    if not window.sangbay:
        # Exercise a manual interval, then restore its automatic result.
        window.table.setCurrentCell(0,0)
        eligible = np.setdiff1d(window.results[0].spectrum.eligible,window.results[0].noise_indices)
        window.manual_range()
        window.range_point(int(eligible[2]))
        window.range_point(int(eligible[-3]))
        wait()
        assert window.results[0].manual and window.results[0].fit is not None
        window.reset_auto()
        wait()
        assert not window.results[0].manual
    export_summary(window.results,destination/'Summary',['xlsx','csv','txt'])
    export_plots(window.results,destination/'plots')
    assert len(list((destination/'plots').glob('*.png')))==len(paths)
    from openpyxl import load_workbook
    book = load_workbook(destination/'Summary.xlsx',read_only=True)
    assert book.sheetnames==['Summary'] and book.active.max_row==len(paths)+1
    book.close()
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    window.table.setCurrentCell(0,0)
    app.processEvents()
    window.grab().save(str(destination/'ui_preview.png'))
    report = {'files':len(paths),'status_counts':dict(Counter(r['status'] for r in rows)),
              'files_with_interference':sum(r['noise_excluded']>0 for r in rows),
              'source_hashes_unchanged':True,'manual_range_verified':not window.sangbay,
              'seconds':perf_counter()-start,'results':rows}
    (destination/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'failure.txt').unlink(missing_ok=True)
