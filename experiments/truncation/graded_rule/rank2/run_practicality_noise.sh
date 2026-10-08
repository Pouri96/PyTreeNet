#!/bin/bash
# repeats of a few resident-memory runs (fresh process each) for the measurement-noise estimate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
for k in 1 2 3 4; do
  timeout 900 python3 traj_jobs.py --cells stag_mu0.5 --arms svd_plain --chis 24,48,96 --mode rss --nproc 3 --out results/pareto_rss_repeat$k.json
  timeout 900 python3 traj_jobs.py --cells stag_mu0.5 --arms dmt --chis 48 --mode rss --nproc 3 --out results/pareto_rss_repeat$k.json
  timeout 900 python3 traj_jobs.py --cells stag_mu0.5 --arms spcfc-a1-b8-wf --chis 16,24,48 --mode rss --nproc 3 --out results/pareto_rss_repeat$k.json
  timeout 900 python3 traj_jobs.py --cells stag_mu0.5 --arms spcfc-a2-b1-wf --chis 16 --mode rss --nproc 3 --out results/pareto_rss_repeat$k.json
done
