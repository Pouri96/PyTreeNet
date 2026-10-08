#!/bin/bash
# Test 1 step 2 batches B (gated spcf), C (controls mu=2, inf), D (T=2); each invocation < 28 min, resumable
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
while pgrep -f "pur_bench.py @results/jobs_A" >/dev/null; do sleep 5; done
for b in B C D; do
  timeout 1700 python pur_bench.py @results/jobs_$b.json results/step2${b}.json 3 > results/step2${b}.log 2>&1
done
