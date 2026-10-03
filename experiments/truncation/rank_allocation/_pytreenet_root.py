"""Import first: puts this repository's ``pytreenet`` ahead of any other installed copy.

It walks up from this file to the first folder holding ``pytreenet/core/truncation/
parameter_budget.py``; if none is found, ``sys.path`` is left unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "pytreenet" / "core" / "truncation" / "parameter_budget.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
