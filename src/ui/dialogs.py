from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox, QTableWidget,
                              QTableWidgetItem, QDialogButtonBox, QLabel, QMessageBox,
                              QCheckBox, QLineEdit)
from eis.parser import inspect_text, Mapping


class MappingDialog(QDialog):
    def __init__(self, table, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Сопоставление столбцов TXT / CSV')
        self.resize(760, 470)
        self.table = table
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f'{table.path.name}\nВыберите частоту, Z′, Z″ и знак исходного столбца.'))
        form = QFormLayout()
        self.delimiter = QComboBox()
        for label, value in [('Табуляция','\t'), ('Точка с запятой',';'), ('Запятая',','), ('Несколько пробелов','aligned'), ('Пробелы','whitespace')]:
            self.delimiter.addItem(label, value)
        self.delimiter.setCurrentIndex(self.delimiter.findData(table.delimiter))
        self.encoding = QComboBox()
        self.encoding.addItems(['utf-8-sig','cp1251','cp1252'])
        self.encoding.setCurrentText(table.encoding)
        form.addRow('Разделитель', self.delimiter)
        form.addRow('Кодировка', self.encoding)
        self.boxes = [QComboBox() for _ in range(3)]
        for label, box in zip(['Частота','Действительная часть Z′','Мнимая часть'], self.boxes):
            form.addRow(label,box)
        self.sign = QComboBox()
        self.sign.addItems(['Z″ (Im Z)', '−Z″ (−Im Z)'])
        form.addRow('Знак столбца',self.sign)
        self.unit = QComboBox()
        self.unit.addItem('Гц',1.0)
        self.unit.addItem('кГц',1e3)
        self.unit.addItem('МГц',1e6)
        form.addRow('Единица частоты',self.unit)
        layout.addLayout(form)
        self.preview = QTableWidget()
        layout.addWidget(self.preview)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()
        self.delimiter.currentIndexChanged.connect(self.reparse)
        self.encoding.currentIndexChanged.connect(self.reparse)

    def refresh(self):
        t = self.table
        for i, box in enumerate(self.boxes):
            box.clear()
            for column, h in enumerate(t.headers):
                box.addItem(f'{column + 1}: {h}',column)
            box.setCurrentIndex(min(i, len(t.headers)-1))
        if t.mapping:
            for box, index in zip(self.boxes, [t.mapping.frequency,t.mapping.real,t.mapping.imag]):
                box.setCurrentIndex(index)
            self.sign.setCurrentIndex(int(t.mapping.negative_imag))
            self.unit.setCurrentIndex(self.unit.findData(t.mapping.frequency_scale))
        self.preview.setColumnCount(len(t.headers))
        self.preview.setHorizontalHeaderLabels(t.headers)
        self.preview.setRowCount(min(8,len(t.rows)))
        for i,row in enumerate(t.rows[:8]):
            for j,cell in enumerate(row):
                self.preview.setItem(i,j,QTableWidgetItem(cell))
        self.preview.resizeColumnsToContents()

    def reparse(self):
        try:
            self.table = inspect_text(self.table.path, self.delimiter.currentData(), self.encoding.currentText())
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self,'Импорт',str(exc))
            self.delimiter.blockSignals(True)
            self.encoding.blockSignals(True)
            self.delimiter.setCurrentIndex(self.delimiter.findData(self.table.delimiter))
            self.encoding.setCurrentText(self.table.encoding)
            self.delimiter.blockSignals(False)
            self.encoding.blockSignals(False)

    def validate(self):
        indices = [box.currentData() for box in self.boxes]
        if None in indices or len(set(indices)) != 3:
            QMessageBox.warning(self,'Импорт','Выберите три разных столбца.')
            return
        self.mapping = Mapping(*indices, bool(self.sign.currentIndex()), self.unit.currentData())
        self.accept()


class ExportDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle('Экспорт результатов')
        layout = QVBoxLayout(self)
        self.name = QLineEdit('Summary')
        layout.addWidget(QLabel('Имя таблицы'))
        layout.addWidget(self.name)
        self.formats = {}
        for fmt in ('xlsx','csv','txt'):
            box = QCheckBox(fmt.upper())
            box.setChecked(fmt == 'xlsx')
            layout.addWidget(box)
            self.formats[fmt] = box
        self.plots = QCheckBox('Сохранять графики PNG')
        layout.addWidget(self.plots)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
