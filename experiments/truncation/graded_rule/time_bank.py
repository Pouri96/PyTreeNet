"""Time the value and gradient of the lookahead bank on a random MPS.

    PYLIBS=<dir with jax> python time_bank.py ising 12 8 3,6 1e-4 1,2
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time
import numpy as np
import _paths  # noqa: F401
import lookahead as LA
import mfc
import jax
import jax.numpy as jnp

model, N, chi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ms = tuple(int(x) for x in sys.argv[4].split(','))
eps = float(sys.argv[5])
kla = tuple(int(x) for x in sys.argv[6].split(','))
batch = int(sys.argv[7]) if len(sys.argv) > 7 else 60
coarse = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
t0 = time.time()
banks = LA.build_banks(model, N, 0.1, ms, kla, eps, coarse=coarse)
print(f"build {time.time() - t0:.0f}s", {m: dict(n=b.n, W=b.W, Dp=b.Dp, MB=round(sum(w.nbytes for w in b.Wts) / 1e6), lenmed=int(np.median(b.lens))) for m, b in banks.items()}, flush=True)
rng = np.random.default_rng(0)
bonds = [1] + [min(chi, 2 ** min(i, N - i)) for i in range(1, N)] + [1]
Ts = [jnp.asarray(rng.normal(size=(bonds[i], 2, bonds[i + 1])) + 1j * rng.normal(size=(bonds[i], 2, bonds[i + 1]))) for i in range(N)]
arr = {m: LA.bank_args(b) for m, b in banks.items()}


def f(Ts):
    tot = 0.0
    for m, b in banks.items():
        tot = tot + jnp.sum(LA.bank_values(Ts, arr[m], b.W, batch) ** 2)
    return tot


vg = jax.jit(jax.value_and_grad(lambda Ts: f(Ts)))
t0 = time.time()
v, g = vg(Ts)
v.block_until_ready()
print(f"compile+first {time.time() - t0:.1f}s", flush=True)
t0 = time.time()
for _ in range(3):
    v, g = vg(Ts)
    v.block_until_ready()
print(f"value+grad per eval {(time.time() - t0) / 3:.2f}s", flush=True)
