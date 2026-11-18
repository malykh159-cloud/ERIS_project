import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

HEADERS = ['Filename', 'Rs_Ohm', 'Rs_SE_Ohm', 'Rp_Ohm', 'Rp_SE_Ohm',
           'Q_S_s_n', 'Q_SE_S_s_n', 'n', 'n_SE', 'C_eff_F', 'C_eff_SE_F',
           'f_max_Hz', 'f_max_SE_Hz', 'Chi2', 'Points_Used', 'Status']


def export_path(destination, fmt):
    """Append an extension without truncating dots in the user-provided base name."""
    destination = Path(destination)
    return destination.parent / (destination.name + '.' + fmt)


def summary_row(result):
    f = result.fit
    return [result.path.name, f.rs if f else None,
            float(f.se[0]) if f and f.se is not None else None,
            f.rp if f else None, float(f.se[1]) if f and f.se is not None else None,
            f.q if f else None, float(f.se[2]) if f and f.se is not None else None,
            f.n if f else None, float(f.se[3]) if f and f.se is not None else None,
            f.c_eff if f else None,f.c_eff_se if f else None,
            f.f_max if f else None,f.f_max_se if f else None,
            f.chi2 if f else None, len(result.used), result.status]


def export_summary(results, destination, formats):
    """Full parameter precision and SE for all four parameters, identical in all formats."""
    destination = Path(destination)
    rows = [summary_row(r) for r in results]
    paths = []
    for fmt in formats:
        path = export_path(destination, fmt)
        with NamedTemporaryFile(dir=path.parent, suffix='.' + fmt, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            if fmt == 'xlsx':
                wb = Workbook()
                ws = wb.active
                ws.title = 'Summary'
                ws.append(HEADERS)
                for row in rows:
                    ws.append(row)
                    # Filenames are text, even when Excel would otherwise see a formula.
                    ws.cell(ws.max_row, 1).data_type = 's'
                ws.freeze_panes = 'B2'
                ws.auto_filter.ref = ws.dimensions
                for cell in ws[1]:
                    cell.font = Font(bold=True, color='FFFFFF')
                    cell.fill = PatternFill('solid', fgColor='254E70')
                    cell.alignment = Alignment(vertical='center')
                ws.row_dimensions[1].height = 26
                widths = [max(24, min(65, max((len(r[0]) for r in rows), default=24) + 2)),
                          *([19]*(len(HEADERS)-3)),17,28]
                for i, width in enumerate(widths, 1):
                    ws.column_dimensions[get_column_letter(i)].width = width
                for row in ws.iter_rows(min_row=2):
                    for i, cell in enumerate(row):
                        if 1 <= i < len(HEADERS)-2:
                            if cell.value is None:
                                cell.value = 'N/A'
                                cell.alignment = Alignment(horizontal='right')
                            else:
                                cell.number_format = '0.0000000000E+00'
                        if i == len(HEADERS)-1 and cell.value != 'OK':
                            cell.font = Font(color='A45100')
                wb.save(temp_path)
                wb.close()
            else:
                with temp_path.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream, delimiter='\t' if fmt == 'txt' else ',')
                    writer.writerow(HEADERS)
                    for row in rows:
                        writer.writerow(['N/A' if x is None else format(x, '.17g') if isinstance(x, float) else x for x in row])
            os.replace(temp_path, path)
            paths.append(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return paths
