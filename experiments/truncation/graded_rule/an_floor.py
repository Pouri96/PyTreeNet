"""what the fired cuts do at low pressure: tails, residual before/after, discarded weight before/after, per chi"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
model, N, T = 'ising', 12, 4.0
for chi in (24, 32, 40, 48):
    cut = spcfast.SPCFast(model, N, a=2, taus=[1.0], fw=0.0, iters=4)
    M.run_tebd(model, N, chi, int(round(T / 0.1)), 0.1, cut, gates=M.make_gates(model, N, 0.1))
    c = np.array(cut.cand)
    lg = np.array(cut.log)
    print(f'chi={chi} candidates {len(c)} fired {cut.fired}  sum of all tails {c[:, 0].sum():.2e}  max tail {c[:, 0].max():.2e}  '
          f'fired: sum tail {lg[:, 2].sum():.2e}  residual f before {lg[:, 0].sum():.2e} after {lg[:, 1].sum():.2e}  disc after/before {lg[:, 3].sum() / lg[:, 2].sum():.2f}')
