"""Import first in every entry point: this repository's ``pytreenet`` and the ``rule/`` modules.

``rule/`` holds flat modules (``mpsenh``, ``mpsenv``, ``tree_rule``, ``rage_rule``) that import each
other by bare name, so the folder itself goes on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _pytreenet_root  # noqa: F401  (puts this repository's pytreenet ahead of any other copy)

HERE = Path(__file__).resolve().parent
RULE = HERE / "rule"
if str(RULE) not in sys.path:
    sys.path.insert(0, str(RULE))
