| Checkpoint | Family | Episodes | Success (95% CI) | Published | Mean coverage (95% CI) | Params M | Chunk ms (eager, batch 1) | Actions per chunk | ms per control step |
|---|---|---|---|---|---|---|---|---|---|
| `lerobot/diffusion_pusht` | diffusion | 50 | 64.0% [52.0%, 78.0%] | 65.4% | 0.924 [0.878, 0.953] | 262.7 | 73.5 | 8 | 9.19 |
| `lerobot/vqbet_pusht` | vqbet | 50 | 54.0% [40.0%, 68.0%] | 57.0% | 0.851 [0.790, 0.905] | 37.5 | 9.2 | 5 | 1.83 |

Paired check on the 50 eval seeds both blocks ran: this repo 27/50 (54.0%) against the checkpoint's own block 32/50 (64.0%); 29 episodes agree, 8 succeed only here and 13 only there, exact McNemar p = 0.38. Mean max coverage 0.851 here against 0.842 there. The published figure is a 500-episode aggregate (57.0 %), so the two arms differ by 8 successes against 13 -- not a resolved gap at this sample size. Source: `results/vqbet_published.jsonl` (`scripts/vqbet_published.py`).
