import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
model, N, T, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
cut = spcfast.SPCFast(model, N, a=2, taus=[1.0], fw=0.0, iters=4)
M.run_tebd(model, N, chi, int(round(T / 0.1)), 0.1, cut, gates=M.make_gates(model, N, 0.1))
print(model, chi, 'fired', cut.fired, 'rejected by bond', getattr(cut, 'rej', {}), 'alpha chosen', cut.achosen)
