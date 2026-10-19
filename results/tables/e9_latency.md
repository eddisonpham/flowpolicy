| Mode | N | Mean ms | p95 ms | Compile s |
|---|---|---|---|---|
| cudagraph | 1 | 2.29 | 2.30 | 0.0 |
| eager | 1 | 5.36 | 5.93 | 0.0 |
| cudagraph | 2 | 2.81 | 2.82 | 0.0 |
| eager | 2 | 8.37 | 9.13 | 0.0 |
| cudagraph | 4 | 3.86 | 3.87 | 0.0 |
| eager | 4 | 14.84 | 15.59 | 0.0 |
| cudagraph | 8 | 5.98 | 5.99 | 0.0 |
| eager | 8 | 27.30 | 27.93 | 0.0 |
| compile-default | 10 | 13.64 | 14.60 | 63.9 |
| compile-max-autotune | 10 | 5.98 | 6.33 | 133.6 |
| compile-reduce-overhead | 10 | 5.80 | 5.89 | 25.8 |
| cudagraph | 10 | 7.09 | 7.24 | 0.0 |
| eager | 10 | 33.46 | 34.48 | 0.0 |
| cudagraph | 16 | 10.21 | 10.24 | 0.0 |
| eager | 16 | 51.98 | 53.16 | 0.0 |
