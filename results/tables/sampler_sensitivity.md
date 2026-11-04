| Latency ms | n | Fast | Stock | Agree | Fast-only | Stock-only | Delta pp | 95% CI | McNemar | psi | Resolves |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 35 | 40.0% | 62.9% | 15 | 6 | 14 | +22.9 | [+0.0, +45.7] | p = 0.12 | 0.57 | 34.6 pp |
| 100 | 35 | 54.3% | 51.4% | 22 | 7 | 6 | -2.9 | [-22.9, +17.1] | p = 1.00 | 0.37 | 27.9 pp |
| 200 | 35 | 45.7% | 48.6% | 28 | 3 | 4 | +2.9 | [-11.4, +17.1] | p = 1.00 | 0.20 | 20.5 pp |
| **all** | 105 | 46.7% | 54.3% | 65 | 16 | 24 | +7.6 | [-3.8, +19.0] | p = 0.27 | 0.38 | 16.7 pp |

Over 105 paired episodes the two loops disagree on 40 (38% of them, psi): 24 succeed only with the stock loop and 16 only with the fast one, for a +7.6 pp difference in rate, exact McNemar p = 0.27. At this discordance a block of 105 resolves 16.7 pp at 80 % power, which is smaller than that bar, so the gap is a null at that resolution, not an equality. Detecting a gap this size at 80 % power would take 513 episodes per arm at this discordance. Of the 40 discordant episodes that recorded both lengths, 40 ended at a different env step (mean absolute change 71.1), so the disagreement is in the trajectory rather than in the last frame. This is a statement about sampler sensitivity, not about either loop being the better policy: 1e-5 of action rounding is amplified by PushT's contact dynamics, so a block is only reproducible if it pins the loop that produced it.
