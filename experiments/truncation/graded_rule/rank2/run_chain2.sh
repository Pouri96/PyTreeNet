#!/bin/bash
# heis control: SVD scan, then spcf in the usable windows
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
while pgrep -f "pur_bench.py @results/jobs_D" >/dev/null; do sleep 5; done
CELLS="stag:1:plain:1,stag:1:back:1,stag:1:plain:1.5,stag:1:back:1.5,stag:1:plain:2,stag:1:back:2,stag:1:plain:3,stag:1:back:3,stag:inf:plain:1.5,stag:inf:plain:3,dw:1:plain:1.5,dw:1:back:1.5,dw:1:plain:3,dw:1:back:3"
timeout 1700 python pur_bench.py "$CELLS" svd 4,6,8,12,16,24,32,48 results/heis_scan_svd.json 3 10 heis > results/heis_scan_svd.log 2>&1
