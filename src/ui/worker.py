from threading import Event
from PySide6.QtCore import QThread, Signal
from eis.parser import inspect_text, spectrum_from_table
from eis.window_detection import analyze
from eis.types import Result


class AnalysisWorker(QThread):
    completed = Signal(int, object)
    progress = Signal(int, int)
    mapping_needed = Signal(object)

    def __init__(self, jobs, mappings, parent=None):
        super().__init__(parent)
        self.jobs = jobs
        self.mappings = dict(mappings)
        self.mapping_event = Event()
        self.mapping_reply = None

    def cancelled(self):
        return self.isInterruptionRequested()

    def run(self):
        for count, (index, path, spectrum, manual) in enumerate(self.jobs,1):
            if self.cancelled():
                break
            try:
                if spectrum is None:
                    try:
                        table = inspect_text(path)
                        mapping = table.mapping or self.mappings.get(table.signature)
                        if mapping is None:
                            # Reuse explicit delimiter/encoding choices, even if automatic
                            # inspection would choose another candidate representation.
                            for signature,saved in self.mappings.items():
                                _,delimiter,encoding = signature
                                try:
                                    candidate = inspect_text(path,delimiter,encoding)
                                except (ValueError,UnicodeError):
                                    continue
                                if candidate.signature == signature:
                                    table,mapping = candidate,saved
                                    break
                        if mapping is None:
                            self.mapping_reply = None
                            self.mapping_event.clear()
                            self.mapping_needed.emit(table)
                            while not self.mapping_event.wait(0.1):
                                if self.cancelled():
                                    raise InterruptedError()
                            if self.mapping_reply is None:
                                raise ValueError('Сопоставление столбцов отменено пользователем.')
                            table, mapping = self.mapping_reply
                            self.mappings[table.signature] = mapping
                        spectrum = spectrum_from_table(table, mapping)
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        result = Result(path, status='Ошибка импорта', reasons=[str(exc)])
                        self.completed.emit(index,result)
                        self.progress.emit(count,len(self.jobs))
                        continue
                result = analyze(spectrum, manual, self.cancelled)
                self.completed.emit(index,result)
            except InterruptedError:
                break
            except Exception as exc:
                self.completed.emit(index,Result(path,status='Ошибка фитирования',reasons=[str(exc)]))
            self.progress.emit(count,len(self.jobs))
