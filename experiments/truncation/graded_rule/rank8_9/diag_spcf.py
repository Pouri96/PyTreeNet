"""Does the tilt do what it is built for?  Per-cut log of the stock spcf (a=2, taus=(), fw) on a cell: window objective of the SVD cut
vs after the tilt, discarded weight, and the effect on the final-state error classes.

    python rank8_9/diag_spcf.py CELL CHI [fw]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import compress_bench as CB  # noqa: E402
import mps_tools as MT  # noqa: E402
import mpsenh as M  # noqa: E402
from spcfast_ext import SPCFastExt  # noqa: E402


def main():
    name, chi = sys.argv[1], int(sys.argv[2])
    fw = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    c = CB.make_cell(name)
    for ext in (False, True):
        T = MT.dense_to_right_mps(c.v, c.N)
        sp = SPCFastExt('ising', c.N, a=2, taus=(), fw=fw)
        if ext:
            sp.ext = c.ext
        MT.sweep_compress(T, chi, sp)
        print(f'== {name} chi={chi} fw={fw} target={"exact reference" if ext else "pre-cut theta (stock)"}: calls {sp.calls}, fired {sp.fired}, '
              f'rejected {getattr(sp, "rej", {})}')
        print('   b   f_svd(window resid^2)   f_tilt      ratio   discarded: svd -> tilt')
        for fs, fc, ds, dc, b, d, tail in sp.log:
            print(f'   {int(b):<3} {fs:12.4e} {fc:14.4e} {fc / fs:9.3f}    {ds:.3e} -> {dc:.3e}')
        v = MT.to_dense(T)
        m = CB.metrics(c, v)
        vs, _ = CB.run_arm(c, 'svd', chi)
        ms = CB.metrics(c, vs)
        print(f"   final state: infid {m['infid']:.3e} (svd {ms['infid']:.3e}); rms_loc4 {m['rms_loc4']:.3e} (svd {ms['rms_loc4']:.3e}); "
              f"rms_far4 {m['rms_far4']:.3e} (svd {ms['rms_far4']:.3e}); dE {m['dE']:.3e} (svd {ms['dE']:.3e})")


if __name__ == '__main__':
    main()
