"""error anatomy: per-site one-site error (rms over x,y,z), per-bond nn error, SVD vs numpy spc, on one cell
    python an_anatomy.py model N T chi"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import hp_bench
import spcfast
model, N, T, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
extra = {}
for it in sys.argv[5:]:
    k, v = it.split('='); extra[k] = eval(v)
dt = 0.1
ex = hp_bench.reference(model, N, T, dt)
G = M.make_gates(model, N, dt); n = int(round(T / dt))
Ts, _ = M.run_tebd(model, N, chi, n, dt, M.svd_cut, gates=G)
cut = spcfast.SPCFast(model, N, a=2, taus=[1.0], fw=0.0, iters=4, **extra)
Tc, _ = M.run_tebd(model, N, chi, n, dt, cut, gates=G)
a = M.local_obs(ex, N, model)
bs, bc = M.local_obs(M.mps_to_dense(Ts), N, model), M.local_obs(M.mps_to_dense(Tc), N, model)
def per(k, groups):
    return [np.sqrt(np.mean((a[k][i * groups:(i + 1) * groups] - b[k][i * groups:(i + 1) * groups]) ** 2)) for b in (bs, bc) for i in range(len(a[k]) // groups)]
es = np.array(per('single', 3)).reshape(2, -1); en = np.array(per('nn', 9)).reshape(2, -1)
np.set_printoptions(precision=1, linewidth=250)
print(model, N, T, chi, 'fired', cut.fired)
print('site        ', np.arange(N))
print('1-site svd  ', es[0] * 1e3)
print('1-site spc  ', es[1] * 1e3)
print('ratio       ', np.round(es[1] / es[0], 2))
print('bond        ', np.arange(N - 1))
print('nn svd      ', en[0] * 1e3)
print('nn spc      ', en[1] * 1e3)
print('ratio       ', np.round(en[1] / en[0], 2))
ranks = M.ranks(Ts)
print('svd bond dims', ranks)
