"""Per-cut cost of the purification spcf cut against the plain SVD cut, as a function of chi.

One fired cut at the centre bond of an N = 10 purification MPS (d = 4) with every bond at chi, in mixed canonical form, with a
two-site tensor whose singular values decay like a TEBD step (so the cut truncates 4 chi -> chi). The spcf options are the
Test 1/2 defaults (a = 2, iters = 4, taus = 1.0, ks = 1-2-3), with every skip rule off so the cut always fires. Wall time is
the median of several repeats, single BLAS thread; peak memory is from tracemalloc (numpy reports its buffers there).

usage: python cost_scaling.py 8,16,24,32,48,64 OUT.json [reps] [a] [N]   (use N >= 16 so the region outer bonds are bulk bonds at chi)
"""
import _p2  # noqa: F401
import sys, json, time, tracemalloc
import numpy as np
import purlib as P
from spcfpur import SPCFPur


def rand_mps(N, d, chi, rng):
    """Random MPS, bond chi in the bulk (capped by d^min(j, N-j)), left canonical up to the centre, right canonical after."""
    dims = [min(chi, d ** min(j, N - j)) for j in range(N + 1)]
    T = []
    for j in range(N):
        A = rng.normal(size=(dims[j], d, dims[j + 1])) + 1j * rng.normal(size=(dims[j], d, dims[j + 1]))
        T.append(A)
    c = N // 2 - 1
    for j in range(c):                                   # left-canonicalize sites < c
        l, _, r = T[j].shape
        Q, R = np.linalg.qr(T[j].reshape(l * d, r))
        T[j] = Q.reshape(l, d, -1)
        T[j + 1] = np.tensordot(R, T[j + 1], axes=(1, 0))
    for j in range(N - 1, c + 1, -1):                    # right-canonicalize sites > c + 1
        l, _, r = T[j].shape
        Q, R = np.linalg.qr(T[j].reshape(l, d * r).T)
        T[j] = Q.T.reshape(-1, d, r)
        T[j - 1] = np.tensordot(T[j - 1], R.T, axes=(2, 0))
    return T, c


def theta_with_spectrum(l, d, r, rng, decay=0.15):
    """(l, d, d, r) tensor with random singular vectors and singular values exp(-decay * i)."""
    m, n = l * d, d * r
    U, _ = np.linalg.qr(rng.normal(size=(m, min(m, n))) + 1j * rng.normal(size=(m, min(m, n))))
    V, _ = np.linalg.qr(rng.normal(size=(n, min(m, n))) + 1j * rng.normal(size=(n, min(m, n))))
    s = np.exp(-decay * np.arange(min(m, n)) * 32.0 / min(m, n))
    s /= np.linalg.norm(s)
    return ((U * s) @ V.conj().T).reshape(l, d, d, r)


def time_call(f, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def main():
    chis = [int(x) for x in sys.argv[1].split(',')]
    out = sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    a = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    d = 4
    rows = []
    for chi in chis:
        rng = np.random.default_rng(chi)
        T, c = rand_mps(N, d, chi, rng)
        l, r = T[c].shape[0], T[c + 1].shape[2]
        th = theta_with_spectrum(l, d, r, rng)
        cut = SPCFPur('ising', N, a=a, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), f_min=0.0, every=1, eps_min=0.0,
                      pattern='all', rel_skip=0.0)
        cut.start(T)
        cut(th, chi, 'R', None, None, c)                 # warm-up: builds the region F cache (state independent, paid once)
        t_svd = time_call(lambda: P.svd_cut_d(th, chi, 'R'), max(reps, 5))
        f0 = cut.fired
        t_spcf = time_call(lambda: cut(th, chi, 'R', None, None, c), reps)
        fired = cut.fired - f0
        tracemalloc.start()
        P.svd_cut_d(th, chi, 'R')
        m_svd = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        tracemalloc.start()
        cut(th, chi, 'R', None, None, c)
        m_spcf = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        row = dict(chi=chi, a=a, l=l, r=r, svd_med=t_svd[0], svd_min=t_svd[1], spcf_med=t_spcf[0], spcf_min=t_spcf[1],
                   ratio_med=t_spcf[0] / t_svd[0], ratio_min=t_spcf[1] / t_svd[1], fired=fired, reps=reps,
                   mem_svd_MB=m_svd / 2 ** 20, mem_spcf_MB=m_spcf / 2 ** 20, theta_MB=th.nbytes / 2 ** 20,
                   tm={k: round(v, 4) for k, v in cut.tm.items()})
        rows.append(row)
        print(f"chi={chi:4d} svd {t_svd[0]*1e3:9.2f} ms  spcf {t_spcf[0]*1e3:10.1f} ms  ratio {row['ratio_med']:6.1f}"
              f"  fired {fired}/{reps}  mem svd {row['mem_svd_MB']:8.1f} MB  spcf {row['mem_spcf_MB']:9.1f} MB  tm {row['tm']}",
              flush=True)
        json.dump(rows, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
