"""Ratios of the spc cut to plain SVD at the same chi, from results/spc_*.json (the svd arm sits in the same file).

    python spc_table.py
Lower is better. Infidelity is reported, not gated.
"""
import json
import glob
import os

FILES = [('Ising N=12 T=4', 'spc_ising12'), ('DW N=12 T=4', 'spc_dw12'), ('ising2 N=12 T=4', 'spc_ising2_12'),
         ('Heis N=12 T=1.5', 'spc_heis12'), ('Ising N=16 T=4', 'spc_n16_ising'), ('DW N=16 T=5', 'spc_n16_dw'),
         ('ising2 N=16 T=4', 'spc_n16_ising2'), ('Heis N=16 T=1.5', 'spc_n16_heis')]
print(f"{'cell':<18}{'chi':>4}{'arm':<36}{'2-site':>8}{'nn':>7}{'energy':>8}{'infid':>7}{'cpu s':>7}")
for label, stem in FILES:
    path = f'results/{stem}.json'
    if not os.path.exists(path):
        continue
    rows = json.load(open(path))
    base = {r['chi']: r for r in rows if r['arm'] == 'svd'}
    for r in sorted(rows, key=lambda r: (r['chi'], r['arm'])):
        if r['arm'] == 'svd' or r['chi'] not in base:
            continue
        b = base[r['chi']]
        arm = r['arm'].replace(':0.5-1.0-1.5-2.0:1-2-3', '')
        print(f"{label:<18}{r['chi']:>4} {arm:<35}{r['rdm2'] / b['rdm2']:>8.2f}{r['nn_rms'] / b['nn_rms']:>7.2f}"
              f"{r['E_abs'] / b['E_abs']:>8.2f}{r['infid'] / b['infid']:>7.2f}{r.get('cpu', r['wall']):>7.0f}")
