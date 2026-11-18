import hashlib
import re
from pathlib import Path
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtCore import Qt, Signal
from eis.model import impedance
from eis import config as C

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
        self.plotItem.vb.setToolTip('')
        self.plotItem.legend.clear()
        self.result = result
        if not result or not result.spectrum:
            return
        s = result.spectrum
        eligible = s.eligible
        used = result.used
        high = np.flatnonzero(s.frequency > C.MAX_FIT_FREQUENCY_HZ)
        excluded = np.setdiff1d(np.arange(len(s.z)), np.r_[eligible, high])
        classified = np.r_[used, result.noise_indices, result.unstable_indices]
        others = np.setdiff1d(eligible, classified)
        other_label = 'Вне ручного диапазона' if result.manual else 'Автоматически исключённый хвост' if result.fit else 'Экспериментальные точки'
        groups = [(excluded, '#b5b9be', 'x', 'Ниже оси / исключены'),
                  (high, '#7c8795', 'x', 'Выше 100 кГц (исключены)'),
                  (result.noise_indices, '#a546b7', 's', 'Возможные сетевые наводки (исключены)'),
                  (result.unstable_indices, '#9c4130', 'x', 'Нестабильный низкочастотный участок'),
                  (others, '#dc963c' if result.fit else '#8b9bad', 't', other_label),
                  (used, '#176faa', 'o', 'Использованы в fit')]
        for indices, color, symbol, label in groups:
            if len(indices):
                points = pg.ScatterPlotItem(
                    x=s.z.real[indices], y=-s.z.imag[indices], data=s.frequency[indices],
                    symbol=symbol, size=7, brush=color, pen=color, name=label,
                    hoverable=True, hoverSize=10, tip=self.point_tip)
                self.addItem(points)
        if result.fit:
            fit = result.fit
            f = s.frequency[used]
            curve = impedance(np.geomspace(max(f), min(f), 500), *fit.params)
            self.plot(curve.real, -curve.imag, pen=pg.mkPen('#c33b46', width=2), name='Модель в диапазоне fit')
            # Extrapolate relative to model corner, so the intercept remains visible for incomplete arcs.
            corner = fit.f_max
            lower = min(f) * 1e-5
            if corner is not None:
                lower = min(lower, corner * 1e-6)
            lower = max(lower, np.finfo(float).tiny)
            ext = impedance(np.geomspace(min(f), lower, 500), *fit.params)
            self.plot(ext.real, -ext.imag, pen=pg.mkPen('#c33b46', width=2, style=Qt.PenStyle.DashLine), name='Экстраполяция')
            self.plot([fit.rs, fit.rs + fit.rp], [0, 0], pen=None, symbol='d', symbolSize=9, symbolBrush='#5b3888')
            for x, label in [(fit.rs, 'Rs'), (fit.rs + fit.rp, 'Rs + Rp')]:
                text = pg.TextItem(label, color='#5b3888', anchor=(0.5, 0))
                text.setPos(x, 0)
                self.addItem(text)
        self.reset_view()

    def reset_view(self):
        # Screen-sized marker/text padding can retain the previous spectrum's
        # enormous scale. Use finite numerical data bounds to reset deterministically.
        bounds = []
        for item in self.plotItem.items:
            if isinstance(item, (pg.ScatterPlotItem, pg.PlotDataItem)):
                x, y = item.getData()
                if x is None or y is None:
                    continue
                valid = np.isfinite(x) & np.isfinite(y)
                if np.any(valid):
                    bounds.append((np.min(x[valid]), np.max(x[valid]),
                                   np.min(y[valid]), np.max(y[valid])))
        if bounds:
            self.plotItem.vb.setRange(
                xRange=(min(b[0] for b in bounds), max(b[1] for b in bounds)),
                yRange=(min(b[2] for b in bounds), max(b[3] for b in bounds)), padding=.06)

    @staticmethod
    def point_tip(x, y, data):
        return (f'Частота: {float(data):.8g} Гц\n'
                f'Z′: {x:.8g} Ом\n−Z″: {y:.8g} Ом').replace('.', ',')

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
