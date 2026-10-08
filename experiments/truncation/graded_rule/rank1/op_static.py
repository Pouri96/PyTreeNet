"""Test 2: static ceiling at a single snapshot.

Exact |O(t)>> of the dense Heisenberg circuit (cached reference, t in {2, 4}, N = 10) is put into exact right-canonical MPS form and compressed
by ONE left-to-right sweep of cuts to chi.  Arms: svd, spcop (targets from the untruncated two-site tensor at each cut, class (c)), spcop with
one-site targets only (ks=1), spcop with ks=(1,2), and spcop with 3 Gauss-Newton passes (optional upper bound).

    python rank1/op_static.py MODEL N T dt STEPS CHIS OUT
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import op_tebd as O
import op_bench as B
import spcop


def exact_mps(c, N):
    """right-canonical exact MPS of the flat vector c (centre, with the norm, on site 0)."""
    T = [None] * N
    M_ = c.reshape(4 ** (N - 1), 4)              # sites 0..N-2 | site N-1
    for i in range(N - 1, 0, -1):
        U, s, Vh = np.linalg.svd(M_, full_matrices=False)
        keep = int(np.sum(s > 1e-14 * s[0]))
        U, s, Vh = U[:, :keep], s[:keep], Vh[:keep]
        T[i] = Vh.reshape(keep, 4, -1)
        M_ = U * s                                # (rows, keep)
        if i > 1:
            M_ = M_.reshape(M_.shape[0] // 4, 4 * keep)
    T[0] = M_.reshape(1, 4, -1)
    return T


def compress(T0, chi, cut, N):
    T = [t.copy() for t in T0]
    if hasattr(cut, 'start'):
        cut.start(T)
    for b in range(N - 1):
        th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
        A = T[b - 1] if b >= 1 else None
        Bn = T[b + 2] if b + 2 <= N - 1 else None
        if cut is O.svd_cut_op:
            T[b], T[b + 1], _, _ = O.svd_cut_op(th, chi, 'R')
        else:
            T[b], T[b + 1], _, _ = cut(th, chi, 'R', A, Bn, b)
    return T


ARMS = {
    'svd': lambda N: spcop.OpCut(N, diag=True),
    'spcop': lambda N: spcop.SPCOp(N, rel_skip=0.0),
    'spcop_k1': lambda N: spcop.SPCOp(N, ks=(1,), rel_skip=0.0),
    'spcop_k12': lambda N: spcop.SPCOp(N, ks=(1, 2), rel_skip=0.0),
    'spcop_p3': lambda N: spcop.SPCOp(N, rel_skip=0.0, passes=3),
    'spcop_it10': lambda N: spcop.SPCOp(N, rel_skip=0.0, iters=10),
}


def one(model, N, T, dt, step, chi, arm, cex, p1e):
    cut = ARMS[arm](N)
    # rebuild the exact MPS inside the worker (cheap)
    T0 = exact_mps(cex, N)
    t0 = time.time()
    Tc = compress(T0, chi, cut, N)
    wall = time.time() - t0
    c = O.mps_to_dense(Tc)
    c = c / np.linalg.norm(c)
    st = B.snapshot_stats(c, cex, N)
    p1 = O.marginal1(c, N)
    Ce, Ca = O.C_from_p1(p1e), O.C_from_p1(p1)
    reg = B.region_stats(Ca - Ce, Ce)
    out = dict(model=model, N=N, step=step, t=step * dt, chi=chi, arm=arm, wall=wall, maxrank=int(max(O.ranks(Tc))), **st,
               dC=reg, dp1_rms=float(np.sqrt(np.mean((p1 - p1e) ** 2))), Ce=Ce.tolist(), Ca=Ca.tolist())
    if isinstance(cut, spcop.SPCOp):
        out.update(fired=cut.fired, rejected=cut.rejected, skipped=cut.skipped, trunc=cut.trunc,
                   cutlog=np.array(cut.log).tolist() if cut.log else [])
    elif cut.log:
        out['cutlog'] = np.array(cut.log).tolist()
    return out


def _job(a):
    return one(*a)


if __name__ == '__main__':
    model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    steps = [int(x) for x in sys.argv[5].split(',')]
    chis = [int(x) for x in sys.argv[6].split(',')]
    arms = sys.argv[7].split(',')
    out = sys.argv[8]
    nproc = int(sys.argv[9]) if len(sys.argv) > 9 else 2
    ref = B.get_ref(model, N, T, dt)
    jobs = []
    for st in steps:
        cex = ref['snaps'][st]
        cex = cex / np.linalg.norm(cex)
        # sanity: the exact MPS reproduces the vector
        T0 = exact_mps(cex, N)
        assert np.abs(O.mps_to_dense(T0) - cex).max() < 1e-12, 'exact MPS build'
        for chi in chis:
            for arm in arms:
                jobs.append((model, N, T, dt, st, chi, arm, cex, ref['p1'][st]))
    res = []
    t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(_job, jobs):
            res.append(r)
            json.dump(res, open(out, 'w'))
            print(f"done t={r['t']:.1f} chi={r['chi']} {r['arm']} wall={r['wall']:.1f}s infid={r['infid']:.3e} win1={r['win1_rms']:.2e} win2={r['win2_rms']:.2e} ({len(res)}/{len(jobs)}, {time.time() - t0:.0f}s)", flush=True)
