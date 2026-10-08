#!/bin/bash
# EXTENSION (outside the plan's gate grid, flagged as such in the report): larger chi at the cold-gate protocol, to see how the equal-error
# CPU ratio moves with chi for gates that keep r.  rel_skip 0.3 throughout.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=rank7/results
BASE='2:0:4:1.0:1-2-3:0:1'
for eps in 1e-7 1e-6; do
  timeout 1700 python -u rank7/t1_cold.py 64,96 $R/t1x_rs0.3_eps$eps.json 2 "$BASE:$eps:all:0.3" > $R/t1x_rs0.3_eps$eps.txt 2>&1
done
echo done > $R/t1x_done.flag
