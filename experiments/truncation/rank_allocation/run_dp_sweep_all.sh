#!/bin/bash
set -e
# No PYTHONPATH needed: _pytreenet_root.py (imported by cells.py/run.py) finds RAGE-BASE
# PTN's own pytreenet on its own. Run from this folder.
for cell in six_mid six_hard six_wide pyrazine holstein; do
  echo "=== $cell : $(date) ==="
  python dp_sweep.py $cell > logs/dp_sweep_$cell.log 2>&1
  echo "=== $cell done, exit $? : $(date) ==="
done
