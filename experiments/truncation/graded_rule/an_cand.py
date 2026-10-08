"""relation between the discarded-weight tail and the SVD-cut residual f over all candidate cuts"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
for model, N, T, chi in [('ising', 12, 4.0, 8), ('isingdw', 12, 4.0, 8), ('ising2', 12, 4.0, 12), ('heis', 12, 1.5, 12)]:
    cut = spcfast.SPCFast(model, N, a=2, taus=[1.0], fw=0.0, iters=4, f_min=0.0)
    M.run_tebd(model, N, chi, int(round(T / 0.1)), 0.1, cut, gates=M.make_gates(model, N, 0.1))
    c = np.array(cut.cand)
    tail, f = c[:, 0], c[:, 1]
    print(model, 'candidates', len(c))
    for tt in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        m = tail > tt
        print(f'   tail>{tt:.0e}: {m.sum():4d} cuts, share of sum f {f[m].sum() / f.sum():.4f}, fired-by-f>1e-6 among them {np.sum(m & (f > 1e-6))} of {np.sum(f > 1e-6)}')
    sl = np.polyfit(np.log(tail[tail > 1e-12]), np.log(f[tail > 1e-12] + 1e-300), 1)
    print('   log f = %.2f log tail + %.2f' % tuple(sl), ' corr', np.corrcoef(np.log(tail[tail > 1e-12]), np.log(f[tail > 1e-12] + 1e-300))[0, 1].round(3))
