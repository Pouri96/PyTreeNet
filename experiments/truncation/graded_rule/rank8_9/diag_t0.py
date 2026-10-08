"""What carries the H-weighted squared error of an SVD sweep?  (explains f_loc^E vs f_loc in T0)

    python rank8_9/diag_t0.py CELL CHI [CHI ...]  >> rank8_9/results/t0_diag.txt
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


def label(cell, e):
    el = cell.elem
    if hasattr(el, 'names'):
        return str(el.names[e])
    N = el.N
    if e < N * N:
        return f'gamma[{e // N},{e % N}]'
    k = e - N * N
    I, J = divmod(k, el.npair)
    return f'Gamma[{el.pairs[I]},{el.pairs[J]}]'


def main():
    cell = CB.make_cell(sys.argv[1])
    el = cell.elem
    for chi in [int(x) for x in sys.argv[2:]]:
        v, info = CB.run_arm(cell, 'svd', chi)
        R = el.values(v)
        d = R - cell.R_ex
        w = el.coef * d
        w2 = np.abs(w) ** 2
        tot = w2.sum()
        print(f'== {cell.name} chi={chi}: sum|c dR|^2 = {tot:.3e}, dE = {w.real.sum():.3e}')
        order = np.argsort(-w2)
        cum = 0.0
        for e in order[:12]:
            cum += w2[e]
            print(f'   {label(cell, e):<34} span={el.span[e]:<3} c={abs(el.coef[e]):8.3f} dR={d[e].real:+.3e}  c*dR={w[e].real:+.3e}  share={w2[e] / tot:.3f}  cum={cum / tot:.3f}')
        near = el.span <= 4
        n1 = el.kind == 1
        print(f'   shares of sum|c dR|^2: 1-body {w2[n1].sum() / tot:.3f}  2-body {w2[~n1].sum() / tot:.3f}; span<=4 {w2[near].sum() / tot:.3f}; '
              f'1-body AND span<=4 {w2[n1 & near].sum() / tot:.3f}; 2-body AND span<=4 {w2[~n1 & near].sum() / tot:.3f}')
        if cell.is_fermion:
            N = el.N
            dg = np.zeros(len(w2), bool)
            dg[:N * N] = np.eye(N, dtype=bool).ravel()
            print(f'   gamma diagonal (orbital occupations) share {w2[dg].sum() / tot:.3f}; number of elements for 99% of the weight: {int(np.searchsorted(np.cumsum(np.sort(w2)[::-1]) / tot, 0.99)) + 1} of {len(w2)}')
        wr = w.real
        print(f'   signed energy error: span<=4 {wr[near].sum():+.3e}, span>4 {wr[~near].sum():+.3e}, total {wr.sum():+.3e}')


if __name__ == '__main__':
    main()
