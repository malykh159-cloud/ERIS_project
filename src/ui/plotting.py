import hashlib
import re
from pathlib import Path
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtCore import Qt, Signal
from eis.model import impedance

pg.setConfigOptions(antialias=True, background='w', foreground='#25364a')


class NyquistPlot(pg.PlotWidget):
    point_selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setLabel('bottom', "Z′", units='Ω')
        self.setLabel('left', "−Z″", units='Ω')
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.18)
        self.addLegend(offset=(12, 12))
        self.result = None
        self.selecting = False
        self.scene().sigMouseClicked.connect(self.clicked)

    def show_result(self, result):
        self.clear()
        self.plotItem.legend.clear()
        self.result = result
        if not result or not result.spectrum:
            return
        s = result.spectrum
        eligible = s.eligible
        used = result.used
        excluded = np.setdiff1d(np.arange(len(s.z)), eligible)
        others = np.setdiff1d(eligible,np.r_[used,result.noise_indices]) if result.fit else np.setdiff1d(eligible,result.noise_indices)
        other_label = 'Вне ручного диапазона' if result.manual else 'Автоматически исключённый хвост' if result.fit else 'Экспериментальные точки'
        groups = [(excluded, '#b5b9be', 'x', 'Ниже оси / исключены'),
                  (result.noise_indices, '#a546b7', 's', 'Возможные сетевые наводки (исключены)'),
                  (others, '#dc963c' if result.fit else '#8b9bad', 't', other_label),
                  (used, '#176faa', 'o', 'Использованы в fit')]
        for indices, color, symbol, label in groups:
            if len(indices):
                self.plot(s.z.real[indices], -s.z.imag[indices], pen=None, symbol=symbol,
                          symbolSize=7, symbolBrush=color, symbolPen=color, name=label)
        if result.fit:
            fit = result.fit
            f = s.frequency[used]
            curve = impedance(np.geomspace(max(f), min(f), 500), *fit.params)
            self.plot(curve.real, -curve.imag, pen=pg.mkPen('#c33b46', width=2), name='Модель в диапазоне fit')
            # Extrapolate relative to model corner, so the intercept remains visible for incomplete arcs.
            corner = (1 / (fit.rp * fit.params[2])) ** (1 / fit.params[3]) / (2 * np.pi)
            lower = min(min(f) * 1e-5, corner * 1e-6)
            ext = impedance(np.geomspace(min(f), lower, 500), *fit.params)
            self.plot(ext.real, -ext.imag, pen=pg.mkPen('#c33b46', width=2, style=Qt.PenStyle.DashLine), name='Экстраполяция')
            self.plot([fit.rs, fit.rs + fit.rp], [0, 0], pen=None, symbol='d', symbolSize=9, symbolBrush='#5b3888')
            for x, label in [(fit.rs, 'Rs'), (fit.rs + fit.rp, 'Rs + Rp')]:
                text = pg.TextItem(label, color='#5b3888', anchor=(0.5, 0))
                text.setPos(x, 0)
                self.addItem(text)
        self.autoRange()

    def clicked(self, event):
        if not self.selecting or not self.result or not self.result.spectrum or event.button() != Qt.MouseButton.LeftButton:
            return
        s = self.result.spectrum
        candidates = s.eligible
        if not len(candidates):
            return
        # Pick in screen coordinates, independent of zoom and axis scaling.
        mouse = event.scenePos()
        positions = [self.plotItem.vb.mapViewToScene(pg.Point(s.z.real[i], -s.z.imag[i])) for i in candidates]
        distances = [(p.x() - mouse.x())**2 + (p.y() - mouse.y())**2 for p in positions]
        self.point_selected.emit(int(candidates[np.argmin(distances)]))


def export_plots(results, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for result in results:
        if not result.spectrum:
            continue
        widget = NyquistPlot()
        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        widget.resize(1100, 750)
        widget.show_result(result)
        widget.plotItem.setTitle(result.path.name)
        widget.show()
        # Offscreen export still needs layout activation; no modal window or event loop.
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', result.path.stem).rstrip('. ')[:100] or 'spectrum'
        digest = hashlib.sha256(str(result.path.resolve()).encode('utf-8')).hexdigest()[:8]
        exporter = pg.exporters.ImageExporter(widget.plotItem)
        exporter.parameters()['width'] = 1600
        exporter.export(str(directory / f'{stem}_{digest}.png'))
        widget.close()
        widget.deleteLater()
