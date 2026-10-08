"""Inject-and-propagate (kick.py) with the window-size weights wk = (w1, w2, w3) of the objective, 3 jobs at a time.
    python run_wk_variants.py     -> results/defect/wk/<model>_<name>_chi<chi>.{json,txt}, then a ratio table
"""
import json, subprocess, sys, os, glob, re
from concurrent.futures import ThreadPoolExecutor
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('results/defect/wk', exist_ok=True)
W = {'w1': [1, 1, 1], 'w10': [10, 10, 1], 'w100': [100, 100, 1], 'w1000': [1000, 1000, 1], 'w100_1only': [100, 1, 1]}
cells = [('heis', 0.8, 12), ('heis', 0.8, 16), ('ising', 3.5, 12)]
jobs = [(m, t0, c, n, w) for (m, t0, c) in cells for n, w in W.items()]


def run(j):
    m, t0, c, n, w = j
    out = f'results/defect/wk/{m}_{n}_chi{c}.json'
    with open(out.replace('.json', '.txt'), 'w') as f:
        subprocess.run([sys.executable, 'kick.py', m, '16', str(t0), str(c), '0.8', out, json.dumps(dict(wk=w))], stdout=f, stderr=subprocess.STDOUT, timeout=1500)


with ThreadPoolExecutor(3) as ex:
    list(ex.map(run, jobs))
rows = []
for f in sorted(glob.glob('results/defect/wk/*.json')):
    m = re.search(r'(\w+)_(w\w+?)_chi(\d+)\.json', f)
    D = {round(d['tprop'], 1): d for d in json.load(open(f))}
    r = lambda t, k: D[t]['spcf'][k] / max(D[t]['svd'][k], 1e-300)
    rows.append((m.group(1), int(m.group(3)), m.group(2), r(0.0, 'single'), r(0.0, 'nn'), r(0.4, 'single'), r(0.4, 'nn'), r(0.8, 'single'), r(0.8, 'nn'), r(0.0, 'infid')))
rows.sort()
print('model chi wk          | tau=0 (1site nn) | tau=.4 (1site nn) | tau=.8 (1site nn) | infid')
for x in rows:
    print(f'{x[0]:5s} {x[1]:3d} {x[2]:11s} | {x[3]:6.2f} {x[4]:6.2f} | {x[5]:6.2f} {x[6]:6.2f} | {x[7]:6.2f} {x[8]:6.2f} | {x[9]:5.2f}')
