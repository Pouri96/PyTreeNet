"""The DP allocation against the bond cap, on final (parameters, infidelity) only.

Each run is compared by what it achieved, never by its target, since a run does not land on its
target exactly. Run ``p`` beats run ``q`` when it is no larger and no less accurate, and better
by the margin in at least one of the two. The front is the runs no other run beats; the ledger
counts an arm's runs that beat a rival run, and those beaten by one.

    python dp_analysis.py              # at a 2% margin
    python dp_analysis.py --margins    # at 1%, 2%, 5% and 10%
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DENSE = RESULTS / "dense"

#: Relative difference below which two runs tie.
MARGIN = 0.02
#: Rows within this ratio of the floor are dropped: the time step limits them, not the rule.
FLOOR_BAND = 1.05

CELLS = ["four", "six_hard", "six_mid", "six_wide", "pyrazine", "holstein"]
BASELINE = "per_bond"
RULE = "dp"


def load(cell: str) -> dict:
    """``{arm: [(final parameters, final infidelity)]}`` from ``results/``, floor rows dropped."""
    dense = json.loads((DENSE / f"{cell}.json").read_text())["arms"]
    store = {BASELINE: json.loads((RESULTS / f"{cell}.json").read_text())["arms"][BASELINE],
             RULE: dense[RULE]}
    flat = []
    for rows in store.values():
        ordered = sorted(rows, key=lambda r: r["params"])
        if len(ordered) >= 2 and ordered[-2]["err"] <= FLOOR_BAND * ordered[-1]["err"]:
            flat.append(ordered[-1]["err"])
    floor = min(flat) if flat else 0.0
    return {a: [(r["params"], r["err"]) for r in rows if r["err"] > FLOOR_BAND * floor]
            for a, rows in store.items()}


def beats(p, q, m: float = MARGIN) -> bool:
    """``p`` ended on no more memory and no more infidelity, and is better by ``m`` in one."""
    return (p[0] <= q[0] * (1 + m) and p[1] <= q[1] * (1 + m)
            and (p[0] <= q[0] / (1 + m) or p[1] <= q[1] / (1 + m)))


def ownership(arms: dict, m: float = MARGIN):
    """``(credit per arm, number of distinct front runs)`` over the pooled runs."""
    pool = [(p, e, a) for a, pts in arms.items() for p, e in pts]
    front = [(p, e, a) for p, e, a in pool
             if not any(beats((q, f), (p, e), m) for q, f, o in pool
                        if o != a or (q, f) != (p, e))]
    seen = {}
    for p, e, a in front:
        seen.setdefault((p, e), set()).add(a)
    credit = {a: 0.0 for a in arms}
    for labels in seen.values():
        for a in labels:
            credit[a] += 1.0 / len(labels)
    return credit, len(seen)


def ledger(arms: dict, arm: str, rivals, m: float = MARGIN):
    """``(runs of ``arm`` that beat some rival run, runs beaten by one)``."""
    other = [q for o in rivals if o != arm for q in arms.get(o, [])]
    wins = sum(any(beats(p, q, m) for q in other) for p in arms[arm])
    losses = sum(any(beats(q, p, m) for q in other) for p in arms[arm])
    return wins, losses


def vs_baseline(cells=CELLS, m=MARGIN):
    """The exact allocation pooled with the bond cap, and its ledger against the cap."""
    keys = [BASELINE, RULE]
    print(f"\n{'=' * 84}")
    print(f"  THE EXACT ALLOCATION AGAINST THE BOND CAP    margin {m:.0%}")
    print(f"  Achieved final parameters and final infidelity only.")
    print(f"{'=' * 84}\n")
    print(f"  {'cell':<12}{'front':>6}  " + "".join(f"{k:>10}" for k in keys)
          + f"{'cap beaten':>12}")
    total = {k: [0.0, 0, 0, 0] for k in keys}
    for c in cells:
        a = load(c)
        credit, n_front = ownership(a, m)
        cap_beaten = sum(any(beats(p, q, m) for p in a[RULE]) for q in a[BASELINE])
        print(f"  {c:<12}{n_front:>6}  " + "".join(
            f"{credit[k] / len(a[k]):>10.2f}" if a.get(k) else f"{'--':>10}" for k in keys)
            + f"{f'{cap_beaten}/{len(a[BASELINE])}':>12}")
        for k in keys:
            if not a.get(k):
                continue
            w, l = ledger(a, k, [BASELINE], m)
            total[k][0] += credit[k]
            total[k][1] += len(a[k])
            total[k][2] += w
            total[k][3] += l
    print(f"\n  {'arm':<12}{'front':>8}{'runs':>7}{'share':>8}"
          f"{'beats cap':>12}{'beaten by cap':>15}")
    for k in keys:
        f, n, w, l = total[k]
        if not n:
            continue
        wins = "--" if k == BASELINE else str(w)
        beaten = "--" if k == BASELINE else str(l)
        print(f"  {k:<12}{f:>8.1f}{n:>7}{f / n:>8.2f}{wins:>12}{beaten:>15}")
    return total


if __name__ == "__main__":
    for margin in ([0.01, 0.02, 0.05, 0.10] if "--margins" in sys.argv else [MARGIN]):
        vs_baseline(m=margin)
