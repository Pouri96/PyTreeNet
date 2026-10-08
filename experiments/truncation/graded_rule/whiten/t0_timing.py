"""T0: time split of spcf (cut.tm) at high chi, plus per-fired-cut cost against np.linalg.svd of the SAME matrix, interleaved.

    python whiten/t0_timing.py model N T chi out.json [taus=1.0] [reps=3]
Single BLAS thread (set below).  One warm-up TEBD run (fills every region cache), then `reps` measured runs on the same cut object.
For every spcf call the SVD of the very matrix it receives is timed (best of 3) immediately before the call, so both see the same
machine load.  Reported: cut.tm fractions over the run (the pre-registered metric: (setup + line) / (svd + setup + solve + line)),
and for fired cuts t_call / t_svd with the component split, as medians over fired cuts and over the repeated runs.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast


class TimedCut(spcfast.SPCFast):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows = []
        self.record = False

    def __call__(self, theta, chi, dirn, A, B, b):
        if not self.record:
            return super().__call__(theta, chi, dirn, A, B, b)
        l, r = theta.shape[0], theta.shape[3]
        Mm = theta.reshape(2 * l, 2 * r)
        ts = []
        for _ in range(3):
            t = time.perf_counter(); np.linalg.svd(Mm, full_matrices=False); ts.append(time.perf_counter() - t)
        tm0 = dict(self.tm); f0 = self.fired
        t = time.perf_counter()
        res = super().__call__(theta, chi, dirn, A, B, b)
        tc = time.perf_counter() - t
        ts2 = []
        for _ in range(3):
            t = time.perf_counter(); np.linalg.svd(Mm, full_matrices=False); ts2.append(time.perf_counter() - t)
        self.rows.append((min(ts + ts2), tc, self.fired > f0, *(self.tm[k] - tm0[k] for k in ('svd', 'setup', 'solve', 'line')), 2 * l, 2 * r))
        return res


def main():
    model, N, T, chi, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    taus = [float(x) for x in sys.argv[6].split('-')] if len(sys.argv) > 6 else [1.0]
    reps = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    dt = 0.1
    n = int(round(T / dt))
    G = M.make_gates(model, N, dt)
    cut = TimedCut(model, N, a=2, fw=0.0, iters=4, taus=taus, ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
    M.run_tebd(model, N, chi, n, dt, cut, gates=G)                      # warm-up (region caches, f32 caches)
    runs = []
    for rep in range(reps):
        cut.rows = []; cut.record = True
        tm0 = dict(cut.tm); f0, c0 = cut.fired, cut.calls
        t0 = time.perf_counter(); M.run_tebd(model, N, chi, n, dt, cut, gates=G); wall_spcf = time.perf_counter() - t0
        cut.record = False
        t0 = time.perf_counter(); M.run_tebd(model, N, chi, n, dt, M.svd_cut, gates=G); wall_svd = time.perf_counter() - t0
        tm = {k: cut.tm[k] - tm0[k] for k in cut.tm}
        rows = np.array(cut.rows, dtype=float)
        fired = rows[:, 2] > 0
        tsvd, tc = rows[:, 0], rows[:, 1]
        comp = rows[fired][:, 3:7].sum(axis=0)
        tot = sum(tm[k] for k in ('svd', 'setup', 'solve', 'line'))
        runs.append(dict(
            fired=int(fired.sum()), calls=len(rows), wall_spcf_run=wall_spcf, wall_svd_run=wall_svd, tm={k: round(v, 4) for k, v in tm.items()},
            frac_run={k: tm[k] / tot for k in ('svd', 'setup', 'solve', 'line')}, setup_plus_line_frac_run=(tm['setup'] + tm['line']) / tot,
            fired_comp_total={k: float(v) for k, v in zip(('svd', 'setup', 'solve', 'line'), comp)},
            fired_setup_plus_line_frac=float((comp[1] + comp[3]) / comp[:4].sum()) if fired.any() else None,
            fired_setup_plus_line_frac_of_nonsvd=float((comp[1] + comp[3]) / comp[1:4].sum()) if fired.any() else None,
            fired_ratio_call_over_svd_median=float(np.median(tc[fired] / tsvd[fired])) if fired.any() else None,
            fired_ratio_sum=float(tc[fired].sum() / tsvd[fired].sum()) if fired.any() else None,
            unfired_ratio_sum=float(tc[~fired].sum() / tsvd[~fired].sum()),
            run_ratio_spcf_over_svd_wall=wall_spcf / wall_svd,
            svd_t_med_fired=float(np.median(tsvd[fired])) if fired.any() else None,
            dim_med_fired=float(np.median(rows[fired][:, 7])) if fired.any() else None))
        print(json.dumps(runs[-1]), flush=True)
    keys = ['setup_plus_line_frac_run', 'fired_setup_plus_line_frac', 'fired_setup_plus_line_frac_of_nonsvd', 'fired_ratio_call_over_svd_median', 'fired_ratio_sum', 'run_ratio_spcf_over_svd_wall']
    summ = {k: float(np.median([r[k] for r in runs])) for k in keys if runs[0][k] is not None}
    summ['frac_run_median'] = {k: float(np.median([r['frac_run'][k] for r in runs])) for k in ('svd', 'setup', 'solve', 'line')}
    summ['fired_median'] = int(np.median([r['fired'] for r in runs]))
    print('MEDIANS', json.dumps(summ))
    json.dump(dict(args=sys.argv[1:], runs=runs, medians=summ), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
