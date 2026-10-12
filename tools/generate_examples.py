"""Clearly labelled synthetic fixtures; never substitute these for measurements."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eis.model import impedance


def generate():
    target = Path(__file__).resolve().parents[1]/'examples'
    target.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    f = np.logspace(5,-2,75)
    clean = impedance(f,12,240,2e-4,0.85)
    noisy = clean + 0.001*abs(clean)*(rng.normal(size=len(f))+1j*rng.normal(size=len(f)))
    tail = noisy.copy()
    tail[48:] += 80*(np.sqrt(f[48]/f[48:])-0.8)*(1-1j)
    incomplete = noisy[:39]
    two = noisy + impedance(f,0,150,0.008,0.8)
    inductive = noisy.copy()
    inductive[:4] = noisy[:4].real+2j
    variants = [('01_clean_arc.txt',f,noisy,'\t',False,False),
                ('02_diffusion_tail.txt',f,tail,';',True,True),
                ('03_incomplete_arc.txt',f[:39],incomplete,',',False,False),
                ('04_two_arcs.txt',f,two,'\t',False,False),
                ('05_inductive.txt',f,inductive,' ',False,False),
                ('06_clean_arc.csv',f,noisy,',',False,False)]
    for name,freq,z,sep,comma,negative in variants:
        lines = ['SYNTHETIC EXAMPLE. Not a SmartStat measurement. Rs=12 ohm; Rp=240 ohm.',
                 sep.join(['Frequency_Hz',"Z'", "-Z''" if negative else "Z''"])]
        for a,b in zip(freq,z):
            values = [a,b.real,-b.imag if negative else b.imag]
            cells = [format(float(v),'.17g').replace('.',',') if comma else format(float(v),'.17g') for v in values]
            lines.append(sep.join(cells))
        (target/name).write_text('\n'.join(lines),encoding='utf-8-sig')


if __name__ == '__main__':
    generate()
