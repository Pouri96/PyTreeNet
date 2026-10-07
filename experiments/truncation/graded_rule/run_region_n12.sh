#!/bin/bash
PY="C:/Users/edpou/Desktop/BUG/.venv/Scripts/python.exe"
$PY -u mfc_bench.py isingdw 12 4.0 0.1 dsvd:2,mfc:2:0.03:3:100,mfcr:2:0.03:3:100:6:2:0.5-1.0-1.5-2.0:1-2 8 results/native_dw_region.json 3 > out_native_dw_region.txt 2>&1
$PY -u mfc_bench.py ising 12 4.0 0.1 mfcr:2:0.03:3:100:6:2:0.5-1.0-1.5-2.0:1-2,mfcr:2:0.03:3:100:7:2:0.5-1.0-1.5-2.0:1-2-3 8 results/native_ising_region.json 2 > out_native_ising_region.txt 2>&1
