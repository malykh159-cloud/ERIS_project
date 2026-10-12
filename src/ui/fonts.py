import os
from pathlib import Path
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


def configure_fonts():
    # Qt's offscreen Windows plugin does not populate the system font database.
    # Load installed Windows fonts for readable Cyrillic/symbols in PNG and UI checks.
    if not QFontDatabase.families():
        directory = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
        for name in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf'):
            path = directory/name
            if path.is_file():
                QFontDatabase.addApplicationFont(str(path))
    QApplication.instance().setFont(QFont('Segoe UI',9))
