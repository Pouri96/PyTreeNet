#!/bin/bash
# T1 of reports/first_moves/rank7_benchmarking.md: gate scan for spcf on Ising N=20, T=8 (existing pareto_spcf.py, unmodified).
# Run from experiments/truncation/graded_rule.  All outputs go to rank7/results/.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=rank7/results
BASE='2:0:4:1.0:1-2-3:0:1'          # a:fw:iters:taus:ks:fmin:every
# SVD timing ladder, before the spcf scan (3 interleaved reps, steady = min)
timeout 1500 python -u pareto_spcf.py ising 20 8.0 24,32,48,64,96,128 - $R/t1_svdtime_pre.json 3 > $R/t1_svdtime_pre.txt 2>&1
for rs in 1e-2 0.1 0.3 0.6; do
  timeout 1700 python -u pareto_spcf.py ising 20 8.0 32 32,48 $R/t1_rs${rs}_eps1e-7.json 1 "$BASE:1e-7:all:$rs" > $R/t1_rs${rs}_eps1e-7.txt 2>&1
done
timeout 1700 python -u pareto_spcf.py ising 20 8.0 32 32,48 $R/t1_rs0.3_eps1e-5.json 1 "$BASE:1e-5:all:0.3" > $R/t1_rs0.3_eps1e-5.txt 2>&1
# SVD timing ladder again, after the scan (brackets the machine load)
timeout 1500 python -u pareto_spcf.py ising 20 8.0 24,32,48,64,96,128 - $R/t1_svdtime_post.json 3 > $R/t1_svdtime_post.txt 2>&1
echo done > $R/t1_done.flag
