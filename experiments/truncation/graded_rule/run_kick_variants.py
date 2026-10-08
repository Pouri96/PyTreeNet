"""Heisenberg inject-and-propagate with objective variants (kick.py), 3 jobs at a time, no shell quoting.
    python run_kick_variants.py          -> results/defect/kickvar/heis_<variant>_chi<chi>.{json,txt}
"""
import json, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('results/defect/kickvar', exist_ok=True)
V = {'tau01': dict(taus=[0.1]), 'tau025': dict(taus=[0.25]), 'tau05': dict(taus=[0.5]), 'tau2': dict(taus=[2.0]), 'static': dict(taus=[]),
     'a3': dict(a=3), 'k12': dict(ks=[1, 2]), 'k1': dict(ks=[1])}
jobs = [(c, n, kw) for c in (12, 16) for n, kw in V.items()]


def run(j):
    c, n, kw = j
    out = f'results/defect/kickvar/heis_{n}_chi{c}.json'
    with open(out.replace('.json', '.txt'), 'w') as f:
        subprocess.run([sys.executable, 'kick.py', 'heis', '16', '0.8', str(c), '0.8', out, json.dumps(kw)], stdout=f, stderr=subprocess.STDOUT, timeout=1500)


with ThreadPoolExecutor(3) as ex:
    list(ex.map(run, jobs))
