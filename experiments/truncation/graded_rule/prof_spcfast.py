"""cProfile of the numpy spc cut on one cell: python prof_spcfast.py model N T chi [a fw iters taus ks fmin]"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, time, cProfile, pstats
import _paths  # noqa: F401
import mpsenh as M
import spcfast
model, N, T, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
a, fw, iters = (int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (2, 0.0, 4)
taus = [float(x) for x in sys.argv[8].split('-')] if len(sys.argv) > 8 else [1.0]
ks = tuple(int(x) for x in sys.argv[9].split('-')) if len(sys.argv) > 9 else (1, 2, 3)
fmin = 0.0
G = M.make_gates(model, N, 0.1); n = int(round(T / 0.1))
cut = spcfast.SPCFast(model, N, a=a, taus=taus, ks=ks, fw=fw, iters=iters, f_min=fmin)
cut._region(2 + 2 * a, False); cut._region(2 + 2 * a - 1, False)
t0 = time.time(); M.run_tebd(model, N, chi, n, 0.1, M.svd_cut, gates=G); tsvd = time.time() - t0
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
M.run_tebd(model, N, chi, n, 0.1, cut, gates=G)
pr.disable(); tsp = time.time() - t0
print(f'svd run {tsvd:.2f}s   spcf run {tsp:.2f}s (profiled)   fired {cut.fired}/{cut.calls} skipped {cut.skipped}  tm {({k: round(v, 2) for k, v in cut.tm.items()})}')
pstats.Stats(pr).sort_stats('tottime').print_stats(14)
