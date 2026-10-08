"""Pareto data for the spc cut on Ising N=12 T=4 dt=0.1, same cell as pareto_n12.py: every chi in its own process, 29 min cap.

    OUTDIR=results/pareto_n12_cold python pareto_spc_n12.py 6     first pass, fills the JAX compile cache (JAXCACHE)
    OUTDIR=results/pareto_n12      python pareto_spc_n12.py 3     timed pass, compile cache warm
Writes <OUTDIR>/spc_<chi>.json and out_pareto_spc_n12.txt. A cell that hits the cap is logged as CAP and left out.
The plain SVD points are the ones already in results/pareto_n12/svd_<chi>.json.
"""
import os
import subprocess
import sys
import time

PY = sys.executable
ARM = 'spc:2:hard:0.01:1:0.5-1.0-1.5-2.0:1-2-3'
CHIS = [int(c) for c in os.environ.get('CHIS', '4,6,8,10,12,16').split(',')]
OUTDIR = os.environ.get('OUTDIR', 'results/pareto_n12')
NPAR = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CAP = 1750

jobs = sorted(CHIS, reverse=True)
running, t0 = [], time.time()
os.makedirs(OUTDIR, exist_ok=True)
log = open('out_pareto_spc_n12.txt', 'a')
log.write(f'--- {OUTDIR} NPAR={NPAR}\n')
while jobs or running:
    while jobs and len(running) < NPAR:
        c = jobs.pop(0)
        out = f'{OUTDIR}/spc_{c}.json'
        if os.path.exists(out):
            continue
        p = subprocess.Popen([PY, '-u', 'mfc_bench.py', 'ising', '12', '4.0', '0.1', ARM, str(c), out, '1'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running.append((c, p, time.time()))
    time.sleep(5)
    for item in list(running):
        c, p, ts = item
        if p.poll() is not None:
            running.remove(item)
            log.write(f'spc chi={c} done rc={p.returncode} {time.time() - ts:.0f}s\n')
            log.flush()
        elif time.time() - ts > CAP:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], capture_output=True)
            running.remove(item)
            log.write(f'spc chi={c} CAP {CAP}s\n')
            log.flush()
log.write(f'ALL DONE {time.time() - t0:.0f}s\n')
log.close()
