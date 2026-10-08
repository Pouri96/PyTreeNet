"""``rule/spcfast.SPCFast`` with an optional external target for the window RDM (the 'error feedback' hook of the plan).

``rule/spcfast.py`` is NOT edited.  This module reads its source, replaces the single line that sets the target of the Gauss-Newton
step, ``ht = hnorm(theta)``, by a call to an overridable method, and executes the patched source as a new module.  If the line is not
found exactly once the import fails, so a changed ``spcfast.py`` cannot silently disable the hook.

    cut = SPCFastExt(model, N, a=2, taus=(), fw=0.0)
    cut.ext = lambda lo, hi: rho_exact_on_sites_lo_to_hi   # D x D, unit trace; omit (None) for the stock behaviour

The stock target is the window RDM of the two-site tensor being cut (including the errors of the cuts already made to its left);
with ``ext`` the cut instead steers the window RDM of the truncated state to the window RDM of the exact reference state.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
import numpy as np

import _paths  # noqa: F401  (puts rule/ on sys.path so that ``import mpsenh`` inside spcfast resolves)

_SRC = Path(__file__).resolve().parent.parent / 'rule' / 'spcfast.py'
_LINE = 'ht = hnorm(theta)'


def _load():
    src = _SRC.read_text()
    assert src.count(_LINE) == 1, f'{_SRC.name}: expected the target line exactly once, found {src.count(_LINE)}'
    src = src.replace(_LINE, 'ht = self._target_ht(hvec, hnorm, theta, lo, hi)')
    mod = types.ModuleType('spcfast_ext_patched')
    mod.__file__ = str(_SRC)
    exec(compile(src, str(_SRC) + ' [patched by rank8_9/spcfast_ext.py]', 'exec'), mod.__dict__)
    return mod


_mod = _load()


class SPCFastExt(_mod.SPCFast):
    ext = None

    def _target_ht(self, hvec, hnorm, theta, lo, hi):
        if self.ext is None:
            return hnorm(theta)
        rho = np.asarray(self.ext(lo, hi))
        return hvec(rho) / float(np.trace(rho).real)
