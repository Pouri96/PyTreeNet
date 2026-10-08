import os, sys, time, cProfile, pstats
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spccut
model, N, T, dt, chi = 'ising', 12, 2.5, 0.1, 6
cut = spccut.SPCCut(model, N, a=2, taus=[0.5, 1.0, 1.5, 2.0], ks=(1, 2, 3), mode='hard', rcond=0.01, passes=1)
G = M.make_gates(model, N, dt)
n = int(round(T / dt))
M.run_tebd(model, N, chi, n, dt, cut, gates=G)          # warm
pr = cProfile.Profile(); pr.enable()
M.run_tebd(model, N, chi, n, dt, cut, gates=G)
pr.disable()
pstats.Stats(pr).sort_stats('cumulative').print_stats(18)
