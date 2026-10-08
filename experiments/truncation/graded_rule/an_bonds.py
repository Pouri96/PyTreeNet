"""per-bond statistics of the fired cuts: count, mean tail, residual before and after, per sweep direction
    python an_bonds.py model N T chi [key=value ...]"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
model, N, T, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
extra = {}
for it in sys.argv[5:]:
    k, v = it.split('='); extra[k] = eval(v)
cut = spcfast.SPCFast(model, N, a=2, taus=[1.0], fw=0.0, iters=4, **extra)
M.run_tebd(model, N, chi, int(round(T / 0.1)), 0.1, cut, gates=M.make_gates(model, N, 0.1))
lg = np.array(cut.log)
print(model, N, T, chi, 'fired', cut.fired)
print('bond  n  mean_tail   sum_f_svd   sum_f_after  ratio')
for b in range(N - 1):
    m = lg[:, 4] == b
    if m.sum():
        print(f'{b:3d} {m.sum():4d}  {lg[m, 6].mean():.2e}  {lg[m, 0].sum():.2e}  {lg[m, 1].sum():.2e}  {lg[m, 1].sum() / lg[m, 0].sum():.2f}')
