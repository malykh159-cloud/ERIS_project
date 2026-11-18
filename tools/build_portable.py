"""Build using project Python and Windows DLLs, without third-party PATH pollution."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distpath', default='dist')
    parser.add_argument('--workpath', default='build')
    parser.add_argument('--specpath', default='.')
    args = parser.parse_args()
    windows = Path(os.environ.get('SystemRoot', 'C:/Windows'))
    # PyInstaller searches PATH for dependencies. Bundled PDF/image tools may
    # provide incompatible ICU/UCRT DLLs with the same names as Windows libraries.
    paths = [Path(sys.executable).parent, Path(sys.base_prefix),
             Path(sys.base_prefix)/'DLLs', windows/'System32', windows,
             windows/'System32/Wbem']
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join(str(p) for p in paths if p.is_dir())
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--windowed',
               '--onedir', '--name', 'EISArcFit', '--paths', str(root/'src')]
    for module in ('PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
                   'PySide6.QtWebEngineQuick', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets',
                   'pandas', 'pytest', 'tkinter', 'matplotlib'):
        command += ['--exclude-module', module]
    command += ['--distpath', args.distpath, '--workpath', args.workpath,
                '--specpath', args.specpath, str(root/'src/main.py')]
    return subprocess.run(command, cwd=root, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
