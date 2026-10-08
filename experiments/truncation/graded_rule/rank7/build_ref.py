"""Build (or load) the dense reference trajectory of the T2 cell and cross-check its last state against _refcache/ising_N20_T8.0_dt0.1.npy."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import ladder_t as L  # noqa: E402
import hp_bench  # noqa: E402

model, N, T = 'ising', 20, 8.0
t0 = time.time()
states, obs = L.ref_trajectory(model, N, T, 0.1, 5)
print('trajectory', states.shape, 'built/loaded in', round(time.time() - t0, 1), 's; times', obs['t'])
last = hp_bench.reference(model, N, T, 0.1)
print('max |traj[-1] - cached reference| =', float(np.abs(np.asarray(states[-1]) - last).max()))
