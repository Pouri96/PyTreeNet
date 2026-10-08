import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
for model, N, T, chi in [('ising', 16, 5.0, 16), ('isingdw', 16, 5.0, 12), ('ising2', 16, 4.0, 12), ('heis', 16, 1.5, 24)]:
    cut = spcfast.SPCFast(model, N, a=2, taus=[1.0], fw=0.0, iters=4)
    M.run_tebd(model, N, chi, int(round(T / 0.1)), 0.1, cut, gates=M.make_gates(model, N, 0.1))
    print(model, 'fired', cut.fired, 'alpha index chosen (1, .5, .25):', cut.achosen)
