#!/bin/bash
PY="C:/Users/edpou/Desktop/BUG/.venv/Scripts/python.exe"
$PY -u mfc_bench.py ising2 12 4.0 0.1 dsvd:2,mfc:2:0.03:3:250,mfcl:2:0.03:3:250:1:1e-9:1-2:1.0:0.5 8 results/native_ising2_N12.json 3 > out_native_ising2.txt 2>&1
$PY -u mfc_bench.py isingdw 12 4.0 0.1 dsvd:2,mfc:2:0.03:3:250,mfcl:2:0.03:3:250:1:1e-9:1-2:1.0:0.5 8 results/native_dw_N12.json 3 > out_native_dw.txt 2>&1
$PY -u mfc_bench.py heis 12 1.5 0.1 dsvd:2,mfc:2:0.03:3:250,mfcl:2:0.03:3:250:1:1e-9:1-2:1.0:0.3 12 results/native_heis_N12.json 3 > out_native_heis.txt 2>&1
