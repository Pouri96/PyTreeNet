"""Out-of-sample gate test: cells with other N / T than the ones the gate constants were fitted on.   python run_gate_oos.py"""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('results/defect/gate_oos', exist_ok=True)
V = ['ref={}', 'g05={"gate":0.5}', 'g1={"gate":1.0}']
cells = [('heis', 12, 1.2, '6,8,10,12,16'), ('heis', 14, 1.0, '8,10,12,16,20'), ('ising', 12, 4.0, '6,8,12,16'), ('ising2', 12, 3.0, '6,8,12,16'),
         ('isingdw', 12, 4.0, '6,8,12,16'), ('ising', 20, 5.0, '16,24,32')]


def run(c):
    m, N, T, chis = c
    out = f'results/defect/gate_oos/{m}_N{N}_T{T:g}.json'
    with open(out.replace('.json', '.txt'), 'w') as f:
        subprocess.run([sys.executable, 'spcf_grid.py', m, str(N), str(T), chis, out] + V, stdout=f, stderr=subprocess.STDOUT, timeout=1700)


with ThreadPoolExecutor(3) as ex:
    list(ex.map(run, cells))
