import os, sys, time, cProfile, pstats, io
os.environ.setdefault('OMP_NUM_THREADS', '1')
import _paths  # noqa
import mpsenh as M
import lookcut
N, chi, T = 12, 8, 4.0
cut = lookcut.LookCut('ising', N, a=1, taus=[1.0], ks=(1, 2, 3), fw=0.1, maxiter=40, skip_tol=1e-10, ftol=1e-8, gtol=1e-6)
run = lambda: M.run_tebd('ising', N, chi, 40, 0.1, cut, gates=M.make_gates('ising', N, 0.1))
run()
pr = cProfile.Profile(); t0 = time.time(); pr.enable(); run(); pr.disable()
print(f"steady pass {time.time()-t0:.1f}s (profiler inflates)")
s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats('tottime'); st.print_stats(14)
for line in s.getvalue().splitlines()[5:28]: print(line[:150].split('pylibs')[-1].split('graded_rule')[-1])
