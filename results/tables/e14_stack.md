| Sampler | Mode | N | Mean chunk ms | p95 ms | Control rate Hz | Share of 100 ms period | Fits 100 ms |
|---|---|---|---|---|---|---|---|
| fast | cudagraph | 4 | 65.2 | 68.0 | 15.35 | 0.65x | yes |
| fast | compile-reduce-overhead | 4 | 65.6 | 67.0 | 15.25 | 0.66x | yes |
| fast | compile-default | 4 | 67.6 | 70.6 | 14.80 | 0.68x | yes |
| stock | eager | 4 | 72.4 | 75.9 | 13.81 | 0.72x | yes |
| fast | eager | 4 | 74.6 | 77.8 | 13.40 | 0.75x | yes |
| fast | compile-reduce-overhead | 10 | 160.2 | 162.8 | 6.24 | 1.60x | no |
| fast | compile-default | 10 | 168.9 | 176.4 | 5.92 | 1.69x | no |
| fast | cudagraph | 10 | 169.1 | 173.2 | 5.91 | 1.69x | no |
| fast | eager | 10 | 183.0 | 191.7 | 5.46 | 1.83x | no |
| stock | eager | 10 | 183.4 | 197.6 | 5.45 | 1.83x | no |

| Arm | Scheduler | N | Workers | Episodes | s/episode | Episodes/hour | s per 100 control steps | Success | Mean coverage |
|---|---|---|---|---|---|---|---|---|---|
| fast-ddim4 | ddim | 4 | 1 | 4 | 2.6 | 1387.2 | 1.19 | 75% | 0.961 |
| fast-ddim4-2w | ddim | 4 | 2 | 4 | 4.7 | 766.5 | 2.11 | 75% | 0.961 |
| stock-ddim10 | ddim | 10 | 1 | 4 | 5.5 | 659.0 | 2.65 | 75% | 0.949 |
| fast-ddim10 | ddim | 10 | 1 | 4 | 5.8 | 622.7 | 2.59 | 75% | 0.956 |
| published | ddpm | 100 | 1 | 4 | 53.1 | 67.8 | 23.89 | 50% | 0.947 |

Same policy, same seeds, two DDIM loops (does the fast sampler change the policy?)

| Loop | Paired episodes | Success | Mean coverage | Mean chunk ms | Episodes agreeing on success |
|---|---|---|---|---|---|
| fast-ddim10 | 20 | 55% | 0.856 | 191.2 | - |
| stock-ddim10 | 20 | 70% | 0.905 | 182.5 | 9/20 |
