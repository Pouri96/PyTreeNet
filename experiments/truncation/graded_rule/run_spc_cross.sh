#!/bin/bash
# spc cut against plain SVD at matched chi, N=12, one invocation per model (each well inside the 30 minute cap)
PY="C:/Users/edpou/Desktop/BUG/.venv/Scripts/python.exe"
A="spc:2:hard:0.01:1:0.5-1.0-1.5-2.0:1-2-3,spc:2:hard:0.001:2:0.5-1.0-1.5-2.0:1-2-3"
timeout 1700 $PY -u mfc_bench.py ising   12 2.5 0.1 svd,spc:2:hard:0.01:1:0.5-1.0-1.5-2.0:1-2-3 6  results/spc_check_ising12.json 2 > out_spc_check.txt 2>&1
timeout 1700 $PY -u mfc_bench.py ising   12 4.0 0.1 svd,$A 8  results/spc_ising12.json 3 > out_spc_ising12.txt 2>&1
timeout 1700 $PY -u mfc_bench.py isingdw 12 4.0 0.1 svd,$A 8  results/spc_dw12.json 3 > out_spc_dw12.txt 2>&1
timeout 1700 $PY -u mfc_bench.py ising2  12 4.0 0.1 svd,$A 12 results/spc_ising2_12.json 3 > out_spc_ising2_12.txt 2>&1
timeout 1700 $PY -u mfc_bench.py heis    12 1.5 0.1 svd,$A 12 results/spc_heis12.json 3 > out_spc_heis12.txt 2>&1
