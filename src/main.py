import sys
import os
if '--smoke-test' in sys.argv or '--check-data' in sys.argv:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtWidgets import QApplication
from ui.window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('EISArcFit')
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    if '--check-data' in sys.argv:
        from measurement_check import run_measurement_check
        from pathlib import Path
        source = Path(sys.argv[sys.argv.index('--check-data')+1])
        destination = Path(sys.argv[sys.argv.index('--output')+1])
        try:
            run_measurement_check(app,window,source,destination)
            window.close()
            return 0
        except Exception:
            import traceback
            destination.mkdir(parents=True,exist_ok=True)
            (destination/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            return 1
    if '--smoke-test' in sys.argv:
        from smoke import run_smoke
        from pathlib import Path
        destination = Path(sys.argv[sys.argv.index('--smoke-test') + 1])
        try:
            run_smoke(app, window, destination)
            window.close()
            return 0
        except Exception:
            import traceback
            destination.mkdir(parents=True, exist_ok=True)
            (destination / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            return 1
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
