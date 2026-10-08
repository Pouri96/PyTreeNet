"""Does the SVD cutoff split near-degenerate singular values?  Fraction of non-trivial cuts with a relative gap
(s[k-1]-s[k])/s[k-1] below 1e-6 / 1e-3, plain SVD TEBD, per chi.   python an_degen.py model N T chi_list"""
import sys
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
dt = 0.1
G = M.make_gates(model, N, dt)
for chi in [int(c) for c in sys.argv[4].split(',')]:
    gaps = []
    def cut(th, chi_, dirn, A, B, b):
        l, r = th.shape[0], th.shape[3]
        s = np.linalg.svd(th.reshape(2 * l, 2 * r), compute_uv=False)
        k = M._rank(s, chi_)
        if k < len(s) and np.sum(s[k:] ** 2) > 1e-7 * np.sum(s ** 2):
            gaps.append((s[k - 1] - s[k]) / s[k - 1])
        return M.svd_cut(th, chi_, dirn)
    M.run_tebd(model, N, chi, int(round(T / dt)), dt, cut, gates=G)
    g = np.array(gaps)
    print(f'chi={chi:3d} cuts={len(g):4d} gap<1e-6: {np.mean(g < 1e-6):.2f}  gap<1e-3: {np.mean(g < 1e-3):.2f}  median gap {np.median(g):.1e}', flush=True)
