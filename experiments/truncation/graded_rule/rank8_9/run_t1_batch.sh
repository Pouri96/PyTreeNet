#!/bin/sh
# T1 arms at matched chi on the non-local proxies; one process per output file (sequential inside a file)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARMS=svd,varfit,spcf,spcf_fw,spcf_ext,spcf_ext_fw
R=rank8_9/results
case "$1" in
  lr)  for c in "lr0.5_nat 3,4,6" "lr1.5_nat 2,3,4" "lr0.5_rand 16,24,32" "lr1.5_rand 24,32,48" "lr3.0_nat 2,3" "lr3.0_rand 24,32,48"; do
         set -- $c; python rank8_9/compress_bench.py t1 $R/t1_lr.json $1 $2 $ARMS; done ;;
  pppsite) for c in "ppp_site_nat 12,16" "ppp_site_fied 12,16" "ppp_site_rand 12,16"; do
         set -- $c; python rank8_9/compress_bench.py t1 $R/t1_pppsite.json $1 $2 $ARMS; done ;;
  pppmo) for c in "ppp_mo_fied $2" "ppp_mo_rand $3"; do set -- $c; python rank8_9/compress_bench.py t1 $R/t1_pppmo.json $1 $2 $ARMS; done ;;
esac
