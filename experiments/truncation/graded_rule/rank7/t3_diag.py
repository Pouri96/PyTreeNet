"""T3 diagnostic: does the spcf step reduce its OWN objective (the static window residual) as much on the kicked circuit as on the continuous quench?

    python rank7/t3_diag.py            # N=20; kicked setting A and B, chi 16 / 24, depth 14;  tilted-Ising quench, chi 16 / 24, T = 5
SPCFast.log holds, per fired cut, (f_svd, f_c, discarded weight of SVD, discarded weight of spcf, bond, direction, tail).  f_c / f_svd is the
fraction of the static window residual that survives the tilt (taus = none), tail is the SVD's discarded weight at the cut.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import _paths  # noqa: F401,E402
import mpsenh as M  # noqa: E402
import floquet as F  # noqa: E402
import spcfast  # noqa: E402

N = 20


def summarize(name, cut, tag=''):
    lg = np.array(cut.log)
    if not len(lg):
        print(f'{name}: nothing fired')
        return
    fs, fc, ds, dc, b, d, tail = lg.T
    ratio = fc / fs
    print(f'{name:34s} calls {cut.calls:5d} fired {cut.fired:5d} ({cut.fired / cut.calls:.2f}) | residual after/before: median {np.median(ratio):.3f} '
          f'mean {np.mean(ratio):.3f} p10 {np.percentile(ratio, 10):.3f} p90 {np.percentile(ratio, 90):.3f} | tail median {np.median(tail):.1e} '
          f'p90 {np.percentile(tail, 90):.1e} | extra discarded weight / tail: median {np.median((dc - ds) / np.maximum(ds, 1e-300)):.2f}')


def kicked(label, chi, depth, taus):
    thJ, thh = F.SETTINGS[label]
    gates = {d: F.make_kicked_gates(N, thJ, thh, d) for d in 'RL'}
    cut = spcfast.SPCFast('ising', N, a=2, fw=0.0, iters=4, taus=taus, ks=(1, 2, 3), f_min=0.0, every=1, eps_min=1e-7, pattern='all', rel_skip=1e-2)
    Tm = F.all_up_mps(N)
    cut.start(Tm)
    for step in range(1, depth + 1):
        d = F.direction_of(step)
        for b in F.sweep_order(N, d):
            th = np.tensordot(Tm[b], Tm[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', gates[d][b], th)
            A = Tm[b - 1] if b >= 1 else None
            B = Tm[b + 2] if b + 2 <= N - 1 else None
            Tm[b], Tm[b + 1], _, _ = cut(th, chi, d, A, B, b)
    return cut


def quench(chi, T, taus):
    dt = 0.1
    cut = spcfast.SPCFast('ising', N, a=2, fw=0.0, iters=4, taus=taus, ks=(1, 2, 3), f_min=0.0, every=1, eps_min=1e-7, pattern='all', rel_skip=1e-2)
    M.run_tebd('ising', N, chi, int(round(T / dt)), dt, cut, gates=M.make_gates('ising', N, dt))
    return cut


if __name__ == '__main__':
    for lab in ('A', 'B'):
        for chi in (16, 24):
            summarize(f'kicked {lab} chi={chi} depth 14 static', kicked(lab, chi, 14, []))
    for chi in (16, 24):
        summarize(f'quench chi={chi} T=5 static', quench(chi, 5.0, []))
        summarize(f'quench chi={chi} T=5 tau=1', quench(chi, 5.0, [1.0]))
