| Checkpoint | Family | Episodes | Success (95% CI) | Published | Mean coverage (95% CI) | Params M | Chunk ms (eager, batch 1) | Actions per chunk | ms per control step | Resolves pp |
|---|---|---|---|---|---|---|---|---|---|---|
| `lerobot/diffusion_pusht` | diffusion | 50 | 64.0% [52.0%, 78.0%] | 65.4% | 0.924 [0.878, 0.953] | 262.7 | 73.5 | 8 | 9.19 | 13.5 |
| `lerobot/vqbet_pusht` | vqbet | 500 | 54.8% [50.6%, 59.2%] | 57.0% | 0.798 [0.773, 0.824] | 37.5 | 9.2 | 5 | 1.83 | 8.6 |

- `lerobot/diffusion_pusht`, 200 shared eval seeds (1000-1199, inside the checkpoint's own block): this repo 104/200 (52.0%) against the checkpoint's own block 138/200 (69.0%); 106 episodes agree, 30 succeed only here and 64 only there, exact McNemar p < 0.001, so the two runs are distinguished at this sample size. The block resolves 13.5 pp at 80 % power (`stats.resolvable_effect`), so a null here is only that coarse. Mean coverage 0.895 here; their block reports 0.915, which is only a lower bound -- gym-pusht clips reward at 1.0, so every success records exactly 0.95 whatever its true coverage. Their published block is 327/500 = 65.4% over 500 episodes, and these shared seeds are a sub-block of it. Source: `results/diffusion_published.jsonl` (`scripts/published_blocks.py`).

Where the diffusion gap sits (episodes binned by the coverage this repo reached; success counted on both sides for the same seeds):

| Our coverage | Episodes | Success here | Success in its block | Mean coverage here | Its coverage, failures only |
|---|---|---|---|---|---|
| 0.00-0.50 | 6 | 0 | 5 | 0.160 | 0.930 |
| 0.50-0.80 | 19 | 0 | 12 | 0.649 | 0.789 |
| 0.80-0.90 | 7 | 0 | 5 | 0.881 | 0.932 |
| 0.90-0.95 | 64 | 0 | 42 | 0.931 | 0.833 |
| 0.95-1.00 | 104 | 104 | 74 | 0.962 | 0.840 |


- `lerobot/vqbet_pusht`, 500 shared eval seeds (100000-100499, inside the checkpoint's own block): this repo 274/500 (54.8%) against the checkpoint's own block 285/500 (57.0%); 261 episodes agree, 114 succeed only here and 125 only there, exact McNemar p = 0.52, so the two runs are not distinguished at this sample size, and the block can only exclude differences above 8.6 pp. The block resolves 8.6 pp at 80 % power (`stats.resolvable_effect`), so a null here is only that coarse. Mean coverage 0.798 here; their block reports 0.776, which is only a lower bound -- gym-pusht clips reward at 1.0, so every success records exactly 0.95 whatever its true coverage. Their published block is 285/500 = 57.0% over 500 episodes, and these shared seeds are a sub-block of it. Source: `results/vqbet_published.jsonl` (`scripts/published_blocks.py`).

- Replication on a disjoint block: 274/500 (54.8%) on seeds 100000-100499 against 108/200 (54.0%) on seeds 100200-100399; no episode is in both blocks, so the samples are independent (two-sided Fisher exact p = 0.87), and the two blocks are not distinguished, so the rate repeats within sampling error. Mean coverage 0.798 against 0.777. Source: `results/vqbet.jsonl` and `results/vqbet_replication.jsonl`.
