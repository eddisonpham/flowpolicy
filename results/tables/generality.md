| Checkpoint | Family | Episodes | Success (95% CI) | Published | Mean coverage (95% CI) | Params M | Chunk ms (eager, batch 1) | Actions per chunk | ms per control step |
|---|---|---|---|---|---|---|---|---|---|
| `lerobot/diffusion_pusht` | diffusion | 50 | 64.0% [52.0%, 78.0%] | 65.4% | 0.924 [0.878, 0.953] | 262.7 | 73.5 | 8 | 9.19 |
| `lerobot/vqbet_pusht` | vqbet | 200 | 54.0% [47.0%, 61.0%] | 57.0% | 0.818 [0.779, 0.853] | 37.5 | 9.2 | 5 | 1.83 |

- `lerobot/diffusion_pusht`, 200 shared eval seeds (1000-1199, inside the checkpoint's own block): this repo 104/200 (52.0%) against the checkpoint's own block 138/200 (69.0%); 106 episodes agree, 30 succeed only here and 64 only there, exact McNemar p < 0.001, so the two runs are distinguished at this sample size. Mean coverage 0.895 here; their block reports 0.915, which is only a lower bound -- gym-pusht clips reward at 1.0, so every success records exactly 0.95 whatever its true coverage. Their published block is 327/500 = 65.4% over 500 episodes, and these shared seeds are a sub-block of it. Source: `results/diffusion_published.jsonl` (`scripts/published_blocks.py`).

Where the diffusion gap sits (episodes binned by the coverage this repo reached; success counted on both sides for the same seeds):

| Our coverage | Episodes | Success here | Success in its block | Mean coverage here | Its coverage, failures only |
|---|---|---|---|---|---|
| 0.00-0.50 | 6 | 0 | 5 | 0.160 | 0.930 |
| 0.50-0.80 | 19 | 0 | 12 | 0.649 | 0.789 |
| 0.80-0.90 | 7 | 0 | 5 | 0.881 | 0.932 |
| 0.90-0.95 | 64 | 0 | 42 | 0.931 | 0.833 |
| 0.95-1.00 | 104 | 104 | 74 | 0.962 | 0.840 |


- `lerobot/vqbet_pusht`, 200 shared eval seeds (100000-100199, inside the checkpoint's own block): this repo 108/200 (54.0%) against the checkpoint's own block 113/200 (56.5%); 107 episodes agree, 44 succeed only here and 49 only there, exact McNemar p = 0.68, so the two runs are not distinguished at this sample size. Mean coverage 0.818 here; their block reports 0.780, which is only a lower bound -- gym-pusht clips reward at 1.0, so every success records exactly 0.95 whatever its true coverage. Their published block is 285/500 = 57.0% over 500 episodes, and these shared seeds are a sub-block of it. Source: `results/vqbet_published.jsonl` (`scripts/published_blocks.py`).

- Replication on a disjoint block: 108/200 (54.0%) on seeds 100000-100199 against 108/200 (54.0%) on seeds 100200-100399; no episode is in both blocks, so the samples are independent (two-sided Fisher exact p = 1.00), and the two blocks are not distinguished, so the rate repeats within sampling error. Mean coverage 0.818 against 0.777. Source: `results/vqbet.jsonl` and `results/vqbet_replication.jsonl`.
