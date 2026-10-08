"""Cost-floor datum: complex SVD of a (2chi x 2chi) matrix vs one complex matmul of the same size, same matrices, interleaved, single thread.
    python whiten/svd_vs_gemm.py out.json
Medians of 15 interleaved repeats.  Shared machine: absolute times are noisy, the ratio is the point."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, time, json
import numpy as np
rng = np.random.default_rng(0)
out = {}
for chi in (16, 32, 64, 128):
    n = 2 * chi
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)); B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    ts, tg, th = [], [], []
    for _ in range(15):
        t = time.perf_counter(); np.linalg.svd(A, full_matrices=False); ts.append(time.perf_counter() - t)
        t = time.perf_counter(); A @ B; tg.append(time.perf_counter() - t)
        t = time.perf_counter(); np.linalg.eigh(A + A.conj().T); th.append(time.perf_counter() - t)
    out[chi] = dict(svd_ms=1e3 * np.median(ts), gemm_ms=1e3 * np.median(tg), eigh_ms=1e3 * np.median(th), svd_over_gemm=float(np.median(ts) / np.median(tg)))
    print(chi, out[chi], flush=True)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
