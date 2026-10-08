import os, sys, time, cProfile, pstats, io
os.environ.setdefault('OMP_NUM_THREADS', '1')
import _paths  # noqa
import mpsenh as M
import obspin
spec = sys.argv[1] if len(sys.argv) > 1 else 'a=1;k=2;wE=1;eps=1e-6'
import pin_bench
cut = pin_bench.make_cut('pin:' + spec, 'ising', 12)
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
M.run_tebd('ising', 12, 8, 40, 0.1, cut, gates=M.make_gates('ising', 12, 0.1))
pr.disable()
print(f"{spec}: total {time.time()-t0:.1f}s (profiled) cuts {cut.calls} fired {cut.fired}")
s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats('tottime'); st.print_stats(10)
import re
for line in s.getvalue().splitlines()[6:20]:
    m = re.match(r'\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+(.*)', line)
    if m: print(f'{m.group(1):>10} tot {m.group(2):>6} cum {m.group(3):>6}  ' + m.group(4).replace(chr(92), '/').split('/')[-1][:70])
