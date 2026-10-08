"""Print the SVD pressure scan: rdm2 of SVD against chi for every cell, and the usable window (1e-3 <= rdm2 <= 1e-1).

    python scan_table.py scan1.json [scan2.json ...]
"""
import sys, json
import numpy as np
rows = []
for f in sys.argv[1:]:
    rows += json.load(open(f))
rows = [r for r in rows if r['arm'] == 'svd']
chis = sorted({r['chi'] for r in rows})
cells = sorted({(r['family'], str(r['mu']), r['T'], r['gauge']) for r in rows}, key=lambda c: (c[0], -1 if c[1] == 'inf' else -float(c[1]), c[2], c[3]))
print(f"{'family':6s} {'mu':>5s} {'T':>3s} {'gauge':5s} | " + ' '.join(f'chi={c:<3d}' .rjust(8) for c in chis) + ' | usable chi (1e-3<=rdm2<=1e-1)')
for c in cells:
    d = {r['chi']: r for r in rows if (r['family'], str(r['mu']), r['T'], r['gauge']) == c}
    vals = [d[x]['rdm2'] if x in d else np.nan for x in chis]
    use = [x for x in chis if x in d and 1e-3 <= d[x]['rdm2'] <= 1e-1]
    print(f"{c[0]:6s} {c[1]:>5s} {c[2]:3g} {c[3]:5s} | " + ' '.join(f'{v:8.1e}' for v in vals) + ' | ' + (','.join(map(str, use)) if use else '-'))
