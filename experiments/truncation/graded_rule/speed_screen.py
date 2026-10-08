"""Accuracy and steady-state cost of cut-local variants in one process. Ising N=12 chi=8 T=4 (the Pareto cell).

    python speed_screen.py
Pass 1 warms the jit caches, pass 2 is timed and scored against the exact solution. SVD reference from results/pareto_n12.
"""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import _paths  # noqa
import mpsenh as M
import hp_bench
import lookcut

N, chi, T, dt, model = 12, 8, 4.0, 0.1, 'ising'
ex = hp_bench.reference(model, N, T, dt)
S = json.load(open('results/pareto_n12/svd_8.json'))[0]
t0 = time.time(); c0 = time.process_time()
M.run_tebd(model, N, chi, 40, dt, M.svd_cut, gates=M.make_gates(model, N, dt))
tsvd, csvd = time.time() - t0, time.process_time() - c0
print(f"SVD steady (1 pass, includes imports warm): wall {tsvd:.2f}s cpu {csvd:.2f}s", flush=True)
CFG = {  # name: (skip, maxiter, every, a, taus)
    'A ref a1 1tau': (1e-10, 40, 1, 1, [1.0]),
    'B maxiter 12': (1e-10, 12, 1, 1, [1.0]),
    'C skip 1e-6': (1e-6, 40, 1, 1, [1.0]),
    'D skip 1e-4': (1e-4, 40, 1, 1, [1.0]),
    'E every 2': (1e-10, 40, 2, 1, [1.0]),
    'F every 4': (1e-10, 40, 4, 1, [1.0]),
    'G skip1e-6 it15 every2': (1e-6, 15, 2, 1, [1.0]),
}
sel = sys.argv[1].split(',') if len(sys.argv) > 1 else list(CFG)
print(f"{'config':<26}{'wall2':>7}{'cpu2':>7}{'x SVD':>7} | {'infid':>6}{'1site':>7}{'nn':>7}{'nnn':>7}   (ratios to SVD) fired skipped")
for name in sel:
    skip, mi, ev, a, taus = CFG[name]
    cut = lookcut.LookCut(model, N, a=a, taus=taus, ks=(1, 2, 3), fw=0.1, maxiter=mi, skip_tol=skip, ftol=1e-8, gtol=1e-6, every=ev)
    run = lambda: M.run_tebd(model, N, chi, 40, dt, cut, gates=M.make_gates(model, N, dt))
    run()
    f1, s1 = cut.fired, cut.skipped
    t0, c0 = time.time(), time.process_time()
    Tm, _ = run()
    w, c = time.time() - t0, time.process_time() - c0
    e = M.errors(ex, M.mps_to_dense(Tm), N, model)
    print(f"{name:<26}{w:7.1f}{c:7.1f}{w / tsvd:7.0f} | {e['infid'] / S['infid']:6.2f}{e['single_rms'] / S['single_rms']:7.2f}"
          f"{e['nn_rms'] / S['nn_rms']:7.2f}{e['nnn_rms'] / S['nnn_rms']:7.2f}   {cut.fired - f1} {cut.skipped - s1}", flush=True)
