| Latency ms | Strategy | Success (95% CI) | Mean coverage | Deadline misses / episode |
|---|---|---|---|---|
| 0 | naive | 56.7% [40.0%, 73.3%] | 0.917 | 29.33 |
| 50 | naive | 60.0% [43.3%, 76.7%] | 0.932 | 26.37 |
| 100 | blend-r2 | 26.7% [13.3%, 43.3%] | 0.875 | 131.70 |
| 100 | naive | 56.7% [40.0%, 73.3%] | 0.897 | 26.97 |
| 100 | naive-r2 | 10.0% [0.0%, 20.1%] | 0.749 | 141.60 |
| 100 | temporal-ensemble-r2 | 53.3% [36.7%, 70.0%] | 0.926 | 120.37 |
| 150 | naive | 46.7% [30.0%, 63.3%] | 0.914 | 31.40 |
| 200 | blend-r2 | 10.0% [0.0%, 20.0%] | 0.864 | 142.80 |
| 200 | naive | 43.3% [26.7%, 60.0%] | 0.915 | 33.00 |
| 200 | naive-r2 | 10.0% [0.0%, 23.3%] | 0.553 | 147.40 |
| 200 | temporal-ensemble-r2 | 43.3% [26.7%, 60.0%] | 0.880 | 120.73 |

Resolution: `blend-r2` and `naive` share 30 episodes at 100 ms and disagree on 17 of them (psi = 0.57), so a same-seed difference in this block is resolvable from 37.0 pp at 80 % power; a smaller one is a null this table cannot read. Each cell is measured at its own n, so its interval is wider than that bar.
