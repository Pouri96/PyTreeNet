"""Truncation rules against plain SVD on final (stored parameters, final error) only.

Same scoring as experiments/truncation/rank_allocation/dp_analysis.py.  A run is two numbers, its
stored parameters and its error at the final time.  Time, modelled cost and the bond cap are never
used.  Run p beats run q when it is no larger and no less accurate, and better by the margin in at
least one of the two.  All arms are pooled; the front is the runs no other run beats; the ledger
counts an arm's runs that beat a baseline run and those beaten by one.  The metric is the final
infidelity, and separately each local-observable error.

    python mps_front.py cells.json            # at a 2% margin
    python mps_front.py cells.json --margins  # at 1%, 2%, 5% and 10%
cells.json maps a cell name to the list of result files that hold its rows.
"""
import json
import sys

MARGIN = 0.02
FLOOR_BAND = 1.05
NUM_FLOOR = 1e-11
METRICS = [('infid', 'infid'), ('single', 'single_rms'), ('nn', 'nn_rms'), ('nnn', 'nnn_rms'), ('E', 'E_abs')]
BASELINE = 'svd'


def load_cell(files):
    """{arm: {chi: row}}; the baseline rows repeated across files are the same runs and are merged."""
    arms = {}
    for f in files:
        for r in json.load(open(f)):
            arms.setdefault(r['arm'], {})[r['chi']] = r
    return arms


def points(arms, key):
    """{arm: [(params, error)]} with numerical-floor rows and near-floor rows dropped."""
    store = {a: sorted(rows.values(), key=lambda r: r['params']) for a, rows in arms.items()}
    flat = [o[-1][key] for o in store.values() if len(o) >= 2 and o[-2][key] <= FLOOR_BAND * o[-1][key]]
    floor = min(flat) if flat else 0.0
    return {a: [(r['params'], r[key]) for r in rows if r[key] > max(NUM_FLOOR, FLOOR_BAND * floor)]
            for a, rows in store.items()}


def beats(p, q, m=MARGIN):
    return (p[0] <= q[0] * (1 + m) and p[1] <= q[1] * (1 + m)
            and (p[0] <= q[0] / (1 + m) or p[1] <= q[1] / (1 + m)))


def ownership(arms, m=MARGIN):
    pool = [(p, e, a) for a, pts in arms.items() for p, e in pts]
    front = [(p, e, a) for p, e, a in pool
             if not any(beats((q, f), (p, e), m) for q, f, o in pool if o != a or (q, f) != (p, e))]
    seen = {}
    for p, e, a in front:
        seen.setdefault((p, e), set()).add(a)
    credit = {a: 0.0 for a in arms}
    for labels in seen.values():
        for a in labels:
            credit[a] += 1.0 / len(labels)
    return credit, len(seen)


def ledger(arms, arm, m=MARGIN):
    """(runs of `arm` that beat a baseline run, runs of `arm` beaten by a baseline run)."""
    base = arms[BASELINE]
    wins = sum(any(beats(p, q, m) for q in base) for p in arms[arm])
    losses = sum(any(beats(q, p, m) for q in base) for p in arms[arm])
    return wins, losses


def report(cells, m):
    names = None
    totals = {}
    print(f"\n{'=' * 100}\n  RULES AGAINST PLAIN SVD   margin {m:.0%}   final (stored parameters, error) only\n{'=' * 100}")
    for cell, files in cells.items():
        arms_rows = load_cell(files)
        rules = sorted(a for a in arms_rows if a != BASELINE)
        names = rules
        print(f"\n  cell {cell}   ({len(arms_rows[BASELINE])} baseline runs)")
        print(f"  {'metric':<8}{'front':>6}  {'svd share':>10}  " + ''.join(f"{r[:22]:>24}" for r in rules))
        print(f"  {'':<8}{'':>6}  {'':>10}  " + ''.join(f"{'share  beats/beaten':>24}" for _ in rules))
        for name, key in METRICS:
            pts = points(arms_rows, key)
            credit, nf = ownership(pts, m)
            row = f"  {name:<8}{nf:>6}  {credit[BASELINE] / max(len(pts[BASELINE]), 1):>10.2f}  "
            for r in rules:
                if not pts.get(r):
                    row += f"{'--':>24}"
                    continue
                w, l = ledger(pts, r, m)
                row += f"{credit[r] / len(pts[r]):>7.2f}  {w:>3d}/{l:<3d} of {len(pts[r]):>2d}".rjust(24)
                t = totals.setdefault((name, r), [0, 0, 0, 0.0])
                t[0] += w
                t[1] += l
                t[2] += len(pts[r])
                t[3] += credit[r]
            print(row)
    print(f"\n  ALL CELLS POOLED   (runs of the rule that beat a baseline run / runs beaten by one / runs)")
    print(f"  {'metric':<8}" + ''.join(f"{r[:22]:>26}" for r in names))
    for name, _ in METRICS:
        row = f"  {name:<8}"
        for r in names:
            t = totals.get((name, r))
            row += (f"{t[0]:>4d} / {t[1]:<4d} of {t[2]:<4d}".rjust(26) if t else f"{'--':>26}")
        print(row)


if __name__ == '__main__':
    cells = json.load(open(sys.argv[1]))
    for margin in ([0.01, 0.02, 0.05, 0.10] if '--margins' in sys.argv else [MARGIN]):
        report(cells, margin)
