"""Bootstrap for the rank2 scripts: parent folder (for _paths, hp_bench, ...) and rule/ on sys.path. Import first."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for p in (str(HERE), str(PARENT)):
    if p not in sys.path:
        sys.path.insert(0, p)
import _paths  # noqa: E402,F401  (puts this repository's pytreenet and rule/ on sys.path)
