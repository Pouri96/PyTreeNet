"""Whole-run comparison of the chunked and unchunked cut (complements test_chunk.py, which compares single cuts to rounding).

Single cuts agree to 1e-14 (float64).  Over a whole TEBD run the states can still differ, because the initial states are symmetric and the
singular values are degenerate: the kept subspace inside a degenerate multiplet is arbitrary, and a 1e-16 difference picks another one.  To tell
this chaos apart from an implementation difference, every pair is compared against the pair (unchunked, unchunked with the initial state
perturbed by 1e-13): if chunked-vs-unchunked differences are of the same size as perturbed-vs-unchunked, the chunked cut is the same algorithm.
Measured: overlap defect 1 - |<psi_1|psi_2>| and the rdm2 error against the dense purification at the end (N = 8, staggered mu = 0.5, plain gauge,
chi = 6, T = 3).

    python test_chunk_runs.py [OUT.json]
"""
import sys
import json
import _p2  # noqa: F401
import numpy as np
import purlib as P
from spcfpur import SPCFPur
from spcfpur_chunk import SPCFPurChunk

N, chi, T = 8, 6, 3.0
ref = P.Reference('ising', N, 'stag', 0.5, 'plain', T)
Gs = P.pur_gates('ising', N, 0.1, 'plain')


def run(cls, kw, a, prec, pert=0.0, seed=0):
    cut = cls('ising', N, a=a, precision=prec, **kw)
    T0 = P.initial_pur_mps(N, 'stag', 0.5)
    if pert:
        rng = np.random.default_rng(seed)
        T0 = [t + pert * (rng.normal(size=t.shape) + 1j * rng.normal(size=t.shape)) for t in T0]
    Tm, _ = P.run_tebd_d(T0, Gs, chi, int(round(T / 0.1)), cut)
    psi = P.mps_to_dense_d(Tm)
    return psi / np.linalg.norm(psi), Tm, cut


def defect(p, q):
    return float(1 - abs(np.vdot(p, q)))


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'results/test_chunk_runs.json'
    rows = []
    for a in (1, 2):
        for prec in ('f64', 'f32'):
            p0, T0_, c0 = run(SPCFPur, {}, a, prec)
            p1, T1_, c1 = run(SPCFPurChunk, dict(blk=5), a, prec)
            p2, T2_, c2 = run(SPCFPurChunk, dict(blk=1), a, prec)
            q0, Tq_, cq = run(SPCFPur, {}, a, prec, pert=1e-13, seed=1)
            q1, Tq1_, cq1 = run(SPCFPur, {}, a, prec, pert=1e-13, seed=2)
            m = {n: P.pur_metrics(ref, t, full=False)['rdm2'] for n, t in (('unchunked', T0_), ('chunk_b5', T1_), ('chunk_b1', T2_), ('unchunked_pert1', Tq_),
                                                                          ('unchunked_pert2', Tq1_))}
            r = dict(a=a, prec=prec, defect_chunk5_vs_unchunked=defect(p1, p0), defect_chunk1_vs_unchunked=defect(p2, p0),
                     defect_pert1_vs_unchunked=defect(q0, p0), defect_pert2_vs_unchunked=defect(q1, p0), defect_pert1_vs_pert2=defect(q0, q1),
                     rdm2=m, fired=(c0.fired, c1.fired, c2.fired, cq.fired, cq1.fired))
            rows.append(r)
            print(json.dumps(r, default=float), flush=True)
    json.dump(rows, open(out, 'w'), indent=1, default=float)
