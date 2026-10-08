"""Pareto data for Ising N=12 T=4 dt=0.1: every (arm, chi) in its own process with a 29 min cap, at most NPAR at once.

    python pareto_n12.py [NPAR]
Writes results/pareto_n12/<arm>_<chi>.json and out_pareto_n12.txt. A cell that hits the cap is logged as CAP and left out.
"""
import os
import subprocess
import sys
import time

PY = sys.executable
ARMS = {
    'svd': 'svd',
    'dsvd': 'dsvd:1.5',
    'lcut': 'lcut:2:0.1:40:0.5-1.0-1.5-2.0:1-2-3',
    'mfcr': 'mfcr:1.5:0.03:3:100:6:2:0.25-0.5-1.0-1.5-2.0:1-2-3',
}
NS = os.environ.get('NSITES', '12')
TF = os.environ.get('TFINAL', '4.0')
DT = os.environ.get('DTSTEP', '0.1')
FAST = os.environ.get('FAST') == '1'
if FAST:
    ARMS['lcut'] = 'lcut:1:0.1:40:1.0:1-2-3:1e-10:1e-8'
    ARMS['mfcr'] = 'mfcr:1.5:0.03:3:30:4:1:1.0:1-2-3'
OUT = os.environ.get('OUTDIR', f'results/pareto_n{NS}' + ('' if (TF, DT) == ('4.0', '0.1') else f'_T{TF}_dt{DT}') + ('_fast' if FAST else ''))
CHIS = [int(c) for c in os.environ.get('CHIS', '4,6,8,10,12,16').split(',')]
SEL = os.environ.get('ARMSEL', 'svd,dsvd,lcut,mfcr').split(',')
NPAR = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CAP = 1750

jobs = [(k, c) for c in CHIS for k in SEL]
running, t0 = [], time.time()
os.makedirs(OUT, exist_ok=True)
log = open('out_pareto_n12.txt', 'a')
while jobs or running:
    while jobs and len(running) < NPAR:
        k, c = jobs.pop(0)
        out = f'{OUT}/{k}_{c}.json'
        if os.path.exists(out):
            continue
        p = subprocess.Popen([PY, '-u', 'mfc_bench.py', 'ising', NS, TF, DT, ARMS[k], str(c), out, '1'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running.append((k, c, p, time.time()))
    time.sleep(5)
    for item in list(running):
        k, c, p, ts = item
        if p.poll() is not None:
            running.remove(item)
            log.write(f'{k} chi={c} done rc={p.returncode} {time.time() - ts:.0f}s\n')
            log.flush()
        elif time.time() - ts > CAP:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], capture_output=True)
            running.remove(item)
            log.write(f'{k} chi={c} CAP {CAP}s\n')
            log.flush()
log.write(f'ALL DONE {time.time() - t0:.0f}s\n')
log.close()
