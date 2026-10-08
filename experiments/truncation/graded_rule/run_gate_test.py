"""Gate test: SVD, ungated spcf and gated spcf (gate = 0.5, 1, 2) on the cells used in this study, errors at the final time.
    python run_gate_test.py      -> results/defect/gate/<cell>.json|txt   (spcf_grid.py, 3 jobs at a time)
"""
import json, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('results/defect/gate', exist_ok=True)
V = ['ref={}', 'g05={"gate":0.5}', 'g1={"gate":1.0}', 'g2={"gate":2.0}']
cells = [('heis', 1.0, '8,12,16,24,32'), ('ising', 5.0, '8,12,16,24'), ('ising2', 3.0, '8,12,16,24'), ('isingdw', 4.0, '8,12,16,24')]


def run(c):
    m, T, chis = c
    out = f'results/defect/gate/{m}_N16_T{T:g}.json'
    with open(out.replace('.json', '.txt'), 'w') as f:
        subprocess.run([sys.executable, 'spcf_grid.py', m, '16', str(T), chis, out] + V, stdout=f, stderr=subprocess.STDOUT, timeout=1700)


with ThreadPoolExecutor(4) as ex:
    list(ex.map(run, cells))
