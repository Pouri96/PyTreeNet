#!/bin/bash
# inject-and-propagate over a chi ladder in both models, 3 jobs at a time
cd "$(dirname "$0")"
{
for c in 8 10 12 16 24; do echo "heis 16 0.8 $c 1.2 results/defect/kick/heis_t08_chi$c.json"; done
for c in 8 12 16 24 32; do echo "ising 16 3.5 $c 1.2 results/defect/kick/ising_t35_chi$c.json"; done
} | xargs -P 3 -L 1 sh -c 'timeout 1500 python kick.py "$@" > "${6%.json}.txt" 2>&1' _
