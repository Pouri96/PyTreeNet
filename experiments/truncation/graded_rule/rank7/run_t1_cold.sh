#!/bin/bash
# T1 supplement: same five gate settings as run_t1.sh, but with the same gate state for the error run and the timing runs (see t1_cold.py).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=rank7/results
BASE='2:0:4:1.0:1-2-3:0:1'
for cfg in "1e-2 1e-7" "0.1 1e-7" "0.3 1e-7" "0.6 1e-7" "0.3 1e-5"; do
  set -- $cfg
  timeout 1700 python -u rank7/t1_cold.py 32,48 $R/t1c_rs$1_eps$2.json 2 "$BASE:$2:all:$1" > $R/t1c_rs$1_eps$2.txt 2>&1
done
echo done > $R/t1c_done.flag
