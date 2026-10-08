### T1. Test 1, ungated spcf, median over the SVD-usable chi window (SVD in the same gauge has 1e-3 <= rdm2 <= 1e-1)

R = SVD in the better of the two gauges / spcf. zz = far ZZ (distance 3, 4) rms spcf / SVD(better gauge). td = physical trace distance spcf / SVD(better gauge). Pass = R(rdm2) >= 2, R(nn) >= 2, zz <= 1.5, td <= 1.10.

| family | mu | T | spcf gauge | chi run | R rdm2 | R nn | R nnn | zz | td | like-for-like rdm2 | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| staggered | 0.25 | 2 | plain | 4,6,8,12 | 2.34 | 2.40 | 1.33 | 1.52 | 1.01 | 3.00 | no |
| staggered | 0.25 | 2 | back | 4,6,8,12 | 1.96 | 2.08 | 1.00 | 1.08 | 1.68 | 2.54 | no |
| staggered | 0.25 | 3 | plain | 6,8,12,16 | 2.52 | 2.32 | 0.99 | 1.10 | 1.17 | 4.32 | no |
| staggered | 0.25 | 3 | back | 4,8,16,32 | 3.23 | 3.05 | 1.39 | 1.56 | 1.44 | 3.70 | no |
| staggered | 0.25 | 4 | plain | 8,12,24,32 | 2.63 | 2.34 | 1.07 | 1.41 | 1.21 | 4.50 | no |
| staggered | 0.25 | 4 | back | 4,8,16,32 | 2.96 | 2.33 | 1.29 | 0.89 | 0.94 | 2.96 | PASS |
| staggered | 0.5 | 2 | plain | 4,6,8 | 2.69 | 2.78 | 1.38 | 0.90 | 0.95 | 2.69 | PASS |
| staggered | 0.5 | 2 | back | 4,6,12,16 | 1.15 | 1.40 | 0.56 | 1.44 | 3.46 | 3.55 | no |
| staggered | 0.5 | 3 | plain | 6,8,12,16 | 4.20 | 4.06 | 1.59 | 0.99 | 0.98 | 4.34 | PASS |
| staggered | 0.5 | 3 | back | 6,12,16,32 | 1.79 | 1.87 | 0.91 | 2.96 | 2.27 | 4.03 | no |
| staggered | 0.5 | 4 | plain | 8,12,24,32 | 4.67 | 4.29 | 1.69 | 1.22 | 0.99 | 5.13 | PASS |
| staggered | 0.5 | 4 | back | 6,12,16,32 | 2.98 | 2.59 | 1.33 | 0.69 | 1.06 | 3.41 | PASS |
| staggered | 1.0 | 2 | plain | 4,6,8 | 2.53 | 2.37 | 1.63 | 0.60 | 0.94 | 2.53 | PASS |
| staggered | 1.0 | 2 | back | 6,8,16,24 | 0.26 | 0.35 | 0.20 | 19.98 | 16.50 | 3.85 | no |
| staggered | 1.0 | 3 | plain | 6,8,12,16 | 4.38 | 4.14 | 2.09 | 0.79 | 0.97 | 4.38 | PASS |
| staggered | 1.0 | 3 | back | 12,16,24,32 | 0.36 | 0.38 | 0.23 | 9.58 | 11.75 | 5.73 | no |
| staggered | 1.0 | 4 | plain | 6,12,16,32 | 4.09 | 3.83 | 2.00 | 0.89 | 0.97 | 4.09 | PASS |
| staggered | 1.0 | 4 | back | 12,16,24,32 | 0.98 | 0.95 | 0.47 | 3.64 | 2.93 | 3.62 | no |
| staggered | 2.0 | 3 | plain | 6,8,12 | 4.16 | 4.40 | 2.38 | 0.38 | 0.92 | 4.16 | PASS |
| staggered | 2.0 | 3 | back | 16,24,32 | 0.01 | 0.01 | 0.01 | 94.59 | 78.11 | 4.89 | no |
| staggered | 2.0 | 4 | plain | 6,8,16,24 | 3.01 | 3.19 | 1.83 | 0.54 | 0.94 | 3.01 | PASS |
| staggered | 2.0 | 4 | back | 16,24,32 | 0.07 | 0.08 | 0.04 | 28.21 | 14.00 | 2.91 | no |
| staggered | inf | 3 | plain | 6,8 | 3.16 | 3.26 | 1.73 | 0.52 | 0.97 | 3.16 | PASS |
| staggered | inf | 3 | back | 16,24,32 | 0.00 | 0.00 | 0.00 | 5977.80 | 1924.01 | 3.55 | no |
| staggered | inf | 4 | plain | 6,8,12,16 | 2.34 | 2.41 | 1.33 | 0.65 | 0.98 | 2.34 | PASS |
| staggered | inf | 4 | back | 16,24,32 | 0.00 | 0.00 | 0.00 | 241.00 | 83.34 | 2.53 | no |
| domain wall | 0.25 | 2 | plain | 4,6,8,12 | 2.30 | 2.28 | 1.33 | 1.60 | 1.01 | 2.96 | no |
| domain wall | 0.25 | 2 | back | 4,6,8,12 | 2.11 | 2.01 | 0.79 | 0.84 | 1.80 | 3.28 | no |
| domain wall | 0.25 | 3 | plain | 6,8,12,16 | 2.81 | 2.54 | 0.92 | 1.44 | 1.14 | 3.89 | no |
| domain wall | 0.25 | 3 | back | 4,8,12,24 | 3.42 | 2.60 | 1.23 | 1.14 | 0.94 | 3.58 | PASS |
| domain wall | 0.25 | 4 | plain | 6,12,16,32 | 1.71 | 1.46 | 0.79 | 1.48 | 1.35 | 3.77 | no |
| domain wall | 0.25 | 4 | back | 4,8,16,32 | 2.71 | 2.45 | 1.36 | 1.03 | 0.93 | 2.71 | PASS |
| domain wall | 0.5 | 2 | plain | 4,6,8 | 2.62 | 2.63 | 1.32 | 1.25 | 0.96 | 2.62 | PASS |
| domain wall | 0.5 | 2 | back | 4,6,12,16 | 1.24 | 1.12 | 0.44 | 2.57 | 3.94 | 3.43 | no |
| domain wall | 0.5 | 3 | plain | 6,8,12,16 | 3.71 | 3.80 | 1.50 | 1.09 | 0.99 | 3.71 | PASS |
| domain wall | 0.5 | 3 | back | 6,12,16,32 | 1.28 | 1.34 | 0.60 | 2.87 | 2.61 | 3.61 | no |
| domain wall | 0.5 | 4 | plain | 6,12,16,32 | 3.00 | 2.65 | 1.56 | 0.88 | 0.99 | 3.00 | PASS |
| domain wall | 0.5 | 4 | back | 6,12,16,32 | 2.21 | 2.01 | 1.26 | 1.09 | 1.08 | 2.58 | PASS |
| domain wall | 1.0 | 2 | plain | 4,6,8 | 2.60 | 2.42 | 1.49 | 0.87 | 0.97 | 2.60 | PASS |
| domain wall | 1.0 | 2 | back | 6,8,12,16 | 0.18 | 0.20 | 0.12 | 17.67 | 15.76 | 2.93 | no |
| domain wall | 1.0 | 3 | plain | 6,8,12,16 | 3.03 | 2.94 | 1.37 | 1.01 | 0.98 | 3.03 | PASS |
| domain wall | 1.0 | 3 | back | 8,12,24,32 | 0.17 | 0.19 | 0.11 | 29.61 | 14.56 | 2.83 | no |
| domain wall | 1.0 | 4 | plain | 8,12,24,32 | 2.50 | 2.43 | 1.56 | 0.96 | 0.98 | 2.50 | PASS |
| domain wall | 1.0 | 4 | back | 12,16,24,32 | 0.29 | 0.33 | 0.24 | 9.10 | 3.20 | 2.38 | no |
| domain wall | 2.0 | 3 | plain | 4,6,8,12 | 2.21 | 2.01 | 1.34 | 0.86 | 0.97 | 2.21 | PASS |
| domain wall | 2.0 | 3 | back | 12,16,24,32 | 0.01 | 0.01 | 0.01 | 322.28 | 91.96 | 3.24 | no |
| domain wall | 2.0 | 4 | plain | 6,8,12,16 | 2.39 | 2.20 | 1.49 | 0.71 | 0.97 | 2.39 | PASS |
| domain wall | 2.0 | 4 | back | 24,32 | 0.02 | 0.02 | 0.01 | 111.84 | 25.59 | 2.84 | no |
| domain wall | inf | 3 | plain | 4,6,8 | 1.47 | 1.37 | 1.13 | 0.87 | 0.99 | 1.47 | no |
| domain wall | inf | 3 | back | 16,24,32 | 0.00 | 0.00 | 0.00 | 107828.84 | 5299.22 | 2.09 | no |
| domain wall | inf | 4 | plain | 6,8,12 | 1.59 | 1.61 | 1.28 | 0.85 | 0.99 | 1.59 | no |
| domain wall | inf | 4 | back | 24,32 | 0.00 | 0.00 | 0.00 | 148076411139.52 | 231042132062.91 | 2.13 | no |

### T1b. Same for the gated arm (gate 1.0), mu <= 1 and controls, plain gauge only (rows with T=3, 4)

| family | mu | T | gauge | R rdm2 | R nn | pass |
|---|---|---|---|---|---|---|
| domain wall | 2.0 | 3 | plain | 2.20 | 2.02 | PASS |
| domain wall | 2.0 | 4 | plain | 2.33 | 2.19 | PASS |
| domain wall | 1.0 | 3 | plain | 2.92 | 2.85 | PASS |
| domain wall | inf | 3 | plain | 1.47 | 1.37 | no |
| domain wall | 1.0 | 4 | plain | 2.10 | 1.96 | no |
| domain wall | inf | 4 | plain | 1.59 | 1.61 | no |
| domain wall | 0.5 | 3 | plain | 3.28 | 3.63 | PASS |
| domain wall | 0.5 | 4 | plain | 2.93 | 2.64 | PASS |
| domain wall | 0.25 | 3 | plain | 2.43 | 2.54 | no |
| domain wall | 0.25 | 4 | plain | 1.70 | 1.45 | no |
| staggered | 2.0 | 3 | plain | 4.16 | 4.32 | PASS |
| staggered | 2.0 | 4 | plain | 3.00 | 3.19 | PASS |
| staggered | 1.0 | 3 | plain | 4.25 | 4.02 | PASS |
| staggered | inf | 3 | plain | 3.16 | 3.26 | PASS |
| staggered | 1.0 | 4 | plain | 4.09 | 3.83 | PASS |
| staggered | inf | 4 | plain | 2.35 | 2.41 | PASS |
| staggered | 0.5 | 3 | plain | 4.20 | 4.06 | PASS |
| staggered | 0.5 | 4 | plain | 3.47 | 3.34 | PASS |
| staggered | 0.25 | 3 | plain | 2.37 | 2.32 | no |
| staggered | 0.25 | 4 | plain | 1.76 | 1.67 | no |

### T0b. Anchor: mu = inf staggered purification (plain gauge) vs the existing d = 2 code, Ising N = 10, T = 3

Entries are d4/d2 ratios of the error (1.0000 = identical). max|d2-d4| is the largest absolute difference over the seven metrics.

| chi | arm | max abs diff | rdm2 | rdm3 | single | nn | nnn | E | infid | fired d2/d4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | svd | 1.1e-14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |
| 6 | spcf | 6.9e-04 | 0.9670 | 0.9846 | 0.9522 | 0.9745 | 1.0413 | 0.9714 | 1.0048 | 145/145 |
| 8 | svd | 3.6e-15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |
| 8 | spcf | 2.1e-09 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 56/56 |
| 12 | svd | 3.6e-14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |
| 12 | spcf | 1.0e-04 | 0.9658 | 0.9993 | 1.0240 | 0.9535 | 1.0127 | 3.5789 | 1.0056 | 33/33 |
| 16 | svd | 4.0e-15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |
| 16 | spcf | 3.0e-12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1/1 |
