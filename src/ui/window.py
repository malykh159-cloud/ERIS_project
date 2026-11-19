import re
from pathlib import Path
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QFileDialog, QSplitter,
    QLabel, QProgressBar, QMessageBox, QDialog, QComboBox, QStackedWidget)
from eis.types import Result
from eis import config as C
from eis.sangbay_model import BOUNDARY_NAME, BOUNDARY_PARAMETERS, TRANSMISSIVE_NAME, TRANSMISSIVE_PARAMETERS
from eis.exporter import export_summary, export_path
from .plotting import NyquistPlot, export_plots
from .dialogs import MappingDialog, ExportDialog
from .worker import AnalysisWorker
from .fonts import configure_fonts


class MainWindow(QMainWindow):
    def __init__(self, *, sangbay=False):
        super().__init__()
        self.sangbay = sangbay
        self.boundary_cpe = sangbay
        self.model_results = {}
        configure_fonts()
        self.setWindowTitle('EISArcFit — Rs и Rp по дуге Найквиста')
        self.resize(1450,850)
        self.setAcceptDrops(True)
        self.results, self.mappings = [], {}
        self.worker = None
        self.cancel_requested = False
        self.first_point = None
        self.range_row = None
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        if sangbay:
            model_row = QHBoxLayout()
            model_row.addWidget(QLabel('Схема'))
            self.model_selector = QComboBox()
            self.model_selector.setAccessibleName('Схема Sangbay')
            self.model_selector.addItems([TRANSMISSIVE_NAME, BOUNDARY_NAME])
            self.model_selector.setCurrentIndex(1)
            model_row.addWidget(self.model_selector)
            model_row.addStretch()
            layout.addLayout(model_row)
        top = QHBoxLayout()
        self.busy_buttons = []
        if sangbay:
            self.busy_buttons.append(self.model_selector)
        for label, action in [('Добавить файл',self.add_files),
                              ('Обработать выбранные',self.process_selected),
                              ('Обработать все',self.process_all),
                              ('Удалить выбранные',self.remove_selected),
                              ('Удалить все',self.clear_files),
                              ('Экспорт',self.export),
                              ('Экспортировать выбранные',self.export_selected)]:
            button = QPushButton(label)
            button.clicked.connect(action)
            top.addWidget(button)
            self.busy_buttons.append(button)
        self.cancel_button = QPushButton('Отменить обработку')
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        top.addWidget(self.cancel_button)
        layout.addLayout(top)
        split = QSplitter()
        headers = ['Файл','Rs, Ω','Rp, Ω','Q, S·sⁿ','n','C_eff, F','f_max, Гц','χ²','Точек','Статус']
        if sangbay:
            headers[8:8] = ['B, √с', 'R_D, Ω']
        self.table = QTableWidget(0,len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeaderItem(3).setToolTip('Параметр CPE Q, единица S·sⁿ (Ω⁻¹·sⁿ). При n ≠ 1 это не ёмкость в фарадах.')
        self.table.horizontalHeaderItem(4).setToolTip('Показатель степени CPE n, безразмерный. Стандартные ошибки всех параметров — только в экспорте.')
        self.table.horizontalHeaderItem(5).setToolTip('Эффективная ёмкость по частоте модельного максимума: C_eff = 1/(2π·Rp·f_max).')
        self.table.horizontalHeaderItem(6).setToolTip('Максимум −Im(Z) модели Rp‖CPE: f_max = (Rp·Q)^(-1/n)/(2π). Это не максимум шумных экспериментальных точек.')
        self.table.horizontalHeaderItem(7).setToolTip('Reduced modulus-weighted chi-square: сумма квадратов нормированных остатков / (2N − 4).')
        if sangbay:
            self.set_sangbay_headers()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.currentCellChanged.connect(self.selection_changed)
        self.table.setAlternatingRowColors(True)
        split.addWidget(self.table)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        navigation = QHBoxLayout()
        controls = QHBoxLayout()
        for label, action, busy in [('Предыдущий',lambda:self.move(-1),False),('Следующий',lambda:self.move(1),False),
                                   ('К предупреждению',self.next_warning,False),
                                   ('Выбрать диапазон вручную',self.manual_range,True),
                                   ('Сбросить ручной / вернуть авто',self.reset_auto,True),
                                   ('Сбросить вид',self.reset_view,False)]:
            if sangbay and busy:
                continue
            b = QPushButton(label)
            b.clicked.connect(action)
            (controls if busy or label == 'Сбросить вид' else navigation).addWidget(b)
            if busy:
                self.busy_buttons.append(b)
        right_layout.addLayout(navigation)
        right_layout.addLayout(controls)
        self.plot = NyquistPlot()
        self.plot.point_selected.connect(self.range_point)
        right_layout.addWidget(self.plot,1)
        self.details = QLabel('Добавьте TXT или CSV-файлы и нажмите «Обработать все».')
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        right_layout.addWidget(self.details)
        split.addWidget(right)
        split.setSizes([790, 660] if sangbay else [660, 790])
        layout.addWidget(split,1)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.statusBar().showMessage(
            'Схема Rs + [CPE ‖ (Rp + Z_D)]. Подгонка по всему допустимому спектру до 100 кГц.'
            if sangbay else
            f'Схема Rs − (Rp || CPE). Автоматический конец дуги — не выше {C.AUTO_BOUNDARY_MAX_HZ:g} Гц.')
        if sangbay:
            self.model_selector.currentIndexChanged.connect(self.switch_sangbay_model)

    def set_sangbay_headers(self):
        parameters = BOUNDARY_PARAMETERS if self.boundary_cpe else TRANSMISSIVE_PARAMETERS
        headers = ['Файл', *(item[0] for item in parameters), 'Статус']
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

    def switch_sangbay_model(self, index):
        boundary_cpe = bool(index)
        if boundary_cpe == self.boundary_cpe:
            return
        if self.worker:
            self.model_selector.blockSignals(True)
            self.model_selector.setCurrentIndex(int(self.boundary_cpe))
            self.model_selector.blockSignals(False)
            return
        row = self.table.currentRow()
        selected = {item.row() for item in self.table.selectedIndexes()}
        self.model_results[self.boundary_cpe] = {r.path: r for r in self.results}
        self.boundary_cpe = boundary_cpe
        cached = self.model_results.get(boundary_cpe, {})
        self.results = [cached.get(r.path) or Result(r.path, spectrum=r.spectrum, sangbay=True,
                                                    boundary_cpe=boundary_cpe) for r in self.results]
        self.table.blockSignals(True)
        try:
            self.set_sangbay_headers()
            for i, result in enumerate(self.results):
                self.update_row(i, result)
            if row >= 0:
                self.table.setCurrentCell(row, 0)
                self.table.clearSelection()
                for i in selected:
                    self.table.selectionModel().select(
                        self.table.model().index(i, 0),
                        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        finally:
            self.table.blockSignals(False)
        self.selection_changed()

    def add_files(self):
        paths,_ = QFileDialog.getOpenFileNames(
            self, 'Выберите спектры SmartStat', '',
            'Спектры TXT и CSV (*.txt *.TXT *.csv *.CSV);;TXT (*.txt *.TXT);;CSV (*.csv *.CSV);;Все файлы (*)')
        self.add_paths(paths)

    def add_paths(self,paths):
        if self.worker:
            return
        existing = {r.path.resolve() for r in self.results}
        for path in paths:
            path = Path(path).resolve()
            if path not in existing and path.is_file():
                existing.add(path)
                self.results.append(Result(path, sangbay=self.sangbay, boundary_cpe=self.boundary_cpe))
                self.table.insertRow(self.table.rowCount())
                self.update_row(len(self.results)-1,self.results[-1])
        if self.results and self.table.currentRow() < 0:
            self.table.setCurrentCell(0,0)

    def update_row(self,index,result):
        self.results[index] = result
        fit = result.fit
        if self.sangbay:
            count = 8 if self.boundary_cpe else 6
            parameters = list(map(float, fit.params)) if fit else [None] * count
            values = [result.path.name, *parameters, result.status]
            display_formats = {i: '.4g' for i in range(1, count + 1)}
        else:
            values = [result.path.name,fit.rs if fit else None,fit.rp if fit else None,
                  fit.q if fit else None,fit.n if fit else None,
                  fit.c_eff if fit else None,fit.f_max if fit else None,
                  fit.chi2 if fit else None,len(result.used),result.status]
            display_formats = {1: '.2f', 2: '.1f', 3: '.2e', 4: '.3g', 5: '.2e',6: '.3g',7: '.1e'}
        for column,value in enumerate(values):
            if value is None:
                text = 'N/A'
            elif column in display_formats:
                # Presentation only: Results and export retain the original floats.
                text = format(value,display_formats[column]).replace('.',',')
            else:
                text = str(value)
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole,value)
            if 1 <= column < len(values) - 1:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            item.setToolTip(str(result.path) + '\n' + '\n'.join(result.reasons) +
                            ('\nПримечания:\n' + '\n'.join(result.notes) if result.notes else ''))
            if column == len(values) - 1 and result.status not in ('OK','Не обработан'):
                item.setForeground(QColor('#a45100'))
            self.table.setItem(index,column,item)
        self.table.resizeColumnsToContents()
        if self.table.currentRow() == index and not self.table.signalsBlocked():
            self.selection_changed()

    def selection_changed(self,*_):
        self.plot.selecting = False
        self.first_point = None
        self.range_row = None
        row = self.table.currentRow()
        if 0 <= row < len(self.results):
            r = self.results[row]
            self.plot.show_result(r)
            detail = [r.path.name, r.status + (' · ручной диапазон' if r.manual else '')]
            if r.sangbay and r.fit:
                detail.append(f'Относительная комплексная СКО: {r.fit.rms_percent:.3f} %.')
            if r.spectrum:
                detail += r.spectrum.notes
                if len(r.used):
                    f = r.spectrum.frequency[r.used]
                    detail.append(f'Диапазон fit: {max(f):.6g} → {min(f):.6g} Гц; {len(r.used)} точек.')
            detail += r.reasons
            if r.notes:
                detail += ['Примечания:', *r.notes]
            self.details.setText('\n'.join(detail))
        else:
            self.plot.show_result(None)

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()},reverse=True)
        for row in rows:
            removed = self.results.pop(row)
            for cached in self.model_results.values():
                cached.pop(removed.path, None)
            self.table.removeRow(row)
        self.selection_changed()

    def clear_files(self):
        self.table.setRowCount(0)
        self.results.clear()
        self.model_results.clear()
        self.selection_changed()

    def process_all(self):
        self.process_rows(range(len(self.results)))

    def process_selected(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            self.statusBar().showMessage('Выделите один или несколько файлов для обработки.')
            return
        self.process_rows(rows)

    def process_rows(self,rows):
        jobs = []
        for i in rows:
            result = self.results[i]
            manual = (int(result.used[0]),int(result.used[-1])) if result.manual and len(result.used) else None
            jobs.append((i,result.path,result.spectrum,manual))
        self.start_jobs(jobs)

    def start_jobs(self,jobs):
        if self.worker or not jobs:
            return
        self.cancel_requested = False
        self.plot.selecting = False
        for button in self.busy_buttons:
            button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0,len(jobs))
        self.progress.setValue(0)
        self.worker = AnalysisWorker(jobs,self.mappings,self, sangbay=self.sangbay,
                                     boundary_cpe=self.boundary_cpe)
        self.worker.completed.connect(self.update_row)
        self.worker.progress.connect(lambda n,total:self.progress.setValue(n))
        self.worker.mapping_needed.connect(self.map_columns)
        self.worker.finished.connect(self.finished)
        self.statusBar().showMessage('Обработка спектров…')
        self.worker.start()

    def map_columns(self,table):
        worker = self.worker
        if not worker:
            return
        if not worker.cancelled():
            dialog = MappingDialog(table,self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                worker.mapping_reply = (dialog.table,dialog.mapping)
                self.mappings[dialog.table.signature] = dialog.mapping
        worker.mapping_event.set()

    def finished(self):
        for button in self.busy_buttons:
            button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        was_cancelled = self.cancel_requested
        self.worker.deleteLater()
        self.worker = None
        self.statusBar().showMessage('Обработка отменена; завершённые результаты сохранены.' if was_cancelled else 'Обработка завершена.')

    def cancel(self):
        if self.worker:
            self.cancel_requested = True
            self.worker.requestInterruption()
            self.worker.mapping_event.set()

    def manual_range(self):
        row = self.table.currentRow()
        if row < 0 or not self.results[row].spectrum:
            return
        self.first_point = None
        self.range_row = row
        self.plot.selecting = True
        self.details.setText('Нажмите на первую, затем на последнюю точку диапазона. Будут использованы все допустимые точки между ними по частоте.')

    def range_point(self,index):
        if self.first_point is None:
            self.first_point = index
            self.details.setText(f'Первая точка выбрана (№ {index + 1}). Нажмите на последнюю точку.')
            return
        row = self.range_row
        r = self.results[row]
        endpoints = (self.first_point,index)
        self.plot.selecting = False
        self.start_jobs([(row,r.path,r.spectrum,endpoints)])

    def reset_auto(self):
        row = self.table.currentRow()
        if row >= 0:
            r = self.results[row]
            self.start_jobs([(row,r.path,r.spectrum,None)])

    def reset_view(self):
        self.plot.reset_view()

    def move(self,step):
        if self.results:
            self.table.setCurrentCell((self.table.currentRow()+step)%len(self.results),0)

    def next_warning(self):
        for step in range(1,len(self.results)+1):
            row = (self.table.currentRow()+step)%len(self.results)
            if self.results[row].status not in ('OK','Не обработан'):
                self.table.setCurrentCell(row,0)
                break

    def export(self):
        self.export_results(self.results, 'Экспорт результатов')

    def export_selected(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            self.statusBar().showMessage('Выделите одну или несколько строк для экспорта.')
            return
        results = [self.results[row] for row in rows]
        self.export_results(results, 'Экспорт выбранных результатов')

    def export_results(self, results, dialog_title):
        if not results:
            return
        dialog = ExportDialog(self)
        dialog.setWindowTitle(dialog_title)
        model_tag = 'boundary-cpe' if self.boundary_cpe else 'transmissive'
        if self.sangbay:
            dialog.name.setText(f'Sangbay-{model_tag}')
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        formats = [f for f,box in dialog.formats.items() if box.isChecked()]
        name = dialog.name.text().strip()
        if not formats or not name or re.search(r'[<>:"/\\|?*\x00-\x1f]',name) or name.endswith(('.', ' ')):
            QMessageBox.warning(self,'Экспорт','Выберите формат и допустимое имя таблицы без расширения.')
            return
        directory = QFileDialog.getExistingDirectory(self,'Куда сохранить результаты')
        if not directory:
            return
        destination = Path(directory)/name
        targets = [export_path(destination,f) for f in formats]
        if any(p.exists() for p in targets):
            if QMessageBox.question(self,'Экспорт','Таблица с таким именем уже существует. Заменить?') != QMessageBox.StandardButton.Yes:
                return
        try:
            paths = export_summary(results,destination,formats)
            if dialog.plots.isChecked():
                plot_dir = Path(directory) / 'plots'
                export_plots(results, plot_dir / model_tag if self.sangbay else plot_dir)
            QMessageBox.information(self,'Экспорт','Сохранено:\n'+'\n'.join(str(p) for p in paths))
        except Exception as exc:
            QMessageBox.warning(self,'Ошибка экспорта',f'Не удалось завершить экспорт: {exc}')

    def dragEnterEvent(self,event):
        if not self.worker and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self,event):
        self.add_paths([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])
        event.acceptProposedAction()

    def closeEvent(self,event):
        if self.worker:
            self.cancel()
            self.statusBar().showMessage('Дождитесь отмены обработки перед закрытием окна.')
            event.ignore()
        else:
            event.accept()


class SensorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('EISArcFit')
        self.resize(1450, 850)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel('Сенсор'))
        self.selector = QComboBox()
        self.selector.setAccessibleName('Сенсор')
        selector_row.addWidget(self.selector)
        selector_row.addStretch()
        layout.addLayout(selector_row)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.workspaces = {}
        for name, sangbay in [('DD-alphasense', False), ('Sangbay', True)]:
            workspace = MainWindow(sangbay=sangbay)
            self.workspaces[name] = workspace
            self.pages.addWidget(workspace)
            self.selector.addItem(name)
        self.selector.currentIndexChanged.connect(self.pages.setCurrentIndex)

    def closeEvent(self, event):
        active = [page for page in self.workspaces.values() if page.worker]
        if active:
            for page in active:
                page.cancel()
                page.statusBar().showMessage('Дождитесь отмены обработки перед закрытием окна.')
            self.pages.setCurrentWidget(active[0])
            self.selector.setCurrentIndex(self.pages.currentIndex())
            event.ignore()
        else:
            event.accept()
