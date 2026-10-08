"""Validation of the w0_fast option of SPCFPurChunk (reference state W0 recomputed in complex64) against the unchunked cut, float32, random cells.

    python test_chunk_wf.py [OUT.json]

Same comparisons as test_chunk.py part A (jvp, vjp, adjoint of the chunked class, returned tensors of one fired cut), block size 2; tolerance of the
float32 rows of the pre-registration (jvp/vjp <= 1e-4, cut tensors <= 1e-3).
"""
import sys
import json
import _p2  # noqa: F401
import numpy as np
from test_chunk import OPT, rel, rand_mps, theta_with_spectrum, SPCFPur
from spcfpur_chunk import SPCFPurChunk

out = sys.argv[1] if len(sys.argv) > 1 else 'results/test_chunk_wf.json'
worst = dict(jvp=0.0, vjp=0.0, cut=0.0, adjoint_chunk=0.0)
n = 0
rr0 = np.random.default_rng(5)
for a in (1, 2):
    for chi in (4, 6):
        rr = np.random.default_rng(100 * chi + a)
        T, c = rand_mps(8, 4, chi, rr)
        for b in (0, 3, 6):
            l, r = T[b].shape[0], T[b + 1].shape[2]
            th = theta_with_spectrum(l, 4, r, rr, decay=0.35)
            for dirn in 'RL':
                k = min(chi, 2 * min(l, r)) if b else chi
                ref = SPCFPur('ising', 8, a=a, precision='f32', **OPT)
                chk = SPCFPurChunk('ising', 8, a=a, precision='f32', blk=2, w0_fast=True, **OPT)
                ref.debug, chk.debug = [], []
                ref.start(T)
                chk.start(T)
                o1 = ref(th.copy(), k, dirn, None, None, b)
                o2 = chk(th.copy(), k, dirn, None, None, b)
                if not ref.debug:
                    continue
                d1, d2 = ref.debug[0], chk.debug[0]
                x = rr0.normal(size=d1['nx'])
                g = rr0.normal(size=d1['nres'])
                worst['jvp'] = max(worst['jvp'], rel(d2['jvp'](x), d1['jvp'](x)))
                worst['vjp'] = max(worst['vjp'], rel(d2['vjp'](g), d1['vjp'](g)))
                worst['cut'] = max(worst['cut'], max(rel(o2[0], o1[0]), rel(o2[1], o1[1])))
                worst['adjoint_chunk'] = max(worst['adjoint_chunk'], float(abs(g @ d2['jvp'](x) - d2['vjp'](g) @ x) / abs(g @ d2['jvp'](x))))
                n += 1
worst['n_cells'] = n
print({k: (f'{v:.2e}' if isinstance(v, float) else v) for k, v in worst.items()})
json.dump(worst, open(out, 'w'), indent=1)
