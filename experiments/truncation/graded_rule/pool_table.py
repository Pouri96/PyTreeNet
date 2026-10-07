"""Ratios of every arm to the delayed-SVD baseline (dsvd:2, same transient) at the same chi, from mfc_bench JSON files.

    python pool_table.py results/native_*.json results/mfcl_*.json
A cell is one (chi) within a file that contains a 'dsvd:2' row. Columns: stored parameters equal by construction, so only
errors are compared. The margin columns report whether the ratio beats 1 by 5 and 10 percent.
"""
import sys, json, glob

METRICS = [('infid', 'infid'), ('rdm2', '2-site'), ('nn_rms', 'nn corr'), ('E_abs', 'energy')]


def load(paths):
    rows = []
    for p in paths:
        for f in glob.glob(p):
            for r in json.load(open(f)):
                rows.append((f.split('/')[-1].replace('.json', ''), r))
    return rows


def main(paths):
    rows = load(paths)
    cells = {}
    for f, r in rows:
        cells.setdefault((f, r['chi']), []).append(r)
    print(f"{'file':<28}{'chi':>4}  {'arm':<58}" + "".join(f"{h:>9}" for _, h in METRICS) + f"{'params':>8}")
    for (f, chi), rs in sorted(cells.items()):
        base = [r for r in rs if r['arm'] == 'dsvd:2']
        if not base:
            continue
        b = base[0]
        for r in sorted(rs, key=lambda r: r['arm']):
            if r['arm'] == 'dsvd:2':
                continue
            ok = r['params'] == b['params']
            print(f"{f:<28}{chi:>4}  {r['arm'][:58]:<58}" + "".join(f"{r[k] / b[k]:>9.2f}" for k, _ in METRICS) + f"{r['params']:>8}" + ('' if ok else '  PARAMS DIFFER'))


if __name__ == '__main__':
    main(sys.argv[1:] or ['results/*.json'])
