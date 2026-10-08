"""Where the spc cut spends its wall time: run the same cell twice with one cut object (jit caches kept)."""
import os, sys, time
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spccut

model, N, T, dt, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), 0.1, int(sys.argv[4])
cut = spccut.SPCCut(model, N, a=2, taus=[0.5, 1.0, 1.5, 2.0], ks=(1, 2, 3), mode='hard', rcond=0.01, passes=1)
G = M.make_gates(model, N, dt)
n = int(round(T / dt))
for rep in range(2):
    c0, f0 = cut.calls, cut.fired
    t0 = time.time()
    M.run_tebd(model, N, chi, n, dt, cut, gates=G)
    print(f"run {rep}: {time.time() - t0:.1f}s calls={cut.calls - c0} fired={cut.fired - f0} keys={len(cut._fun)}", flush=True)
