"""One-time pre-change comparison. Explicit mapping reflects user-confirmed file convention."""
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eis.parser import read_spectrum,Mapping
from eis.window_detection import analyze

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    paths = sorted(Path('data_test').glob('*.txt'))
    destination = Path('outputs/real_check_2026-10-12')
    destination.mkdir(parents=True,exist_ok=True)
    rows = []
    for i,path in enumerate(paths,1):
        start = perf_counter()
        result = analyze(read_spectrum(path,Mapping(1,2,3,True)))
        row = {'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'status':result.status,'rs':result.fit.rs if result.fit else None,
               'rp':result.fit.rp if result.fit else None,'points_used':len(result.used),
               'end_frequency_hz':float(result.spectrum.frequency[result.used[-1]]) if len(result.used) else None,
               'reasons':result.reasons,'seconds':perf_counter()-start}
        rows.append(row)
        (destination/'baseline.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(i,len(paths),path.name,row['status'],row['points_used'],row['end_frequency_hz'],flush=True)
