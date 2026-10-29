| Checkpoint | Family | Episodes | Success (95% CI) | Published | Mean coverage (95% CI) | Params M | Chunk ms (eager, batch 1) | Actions per chunk | ms per control step |
|---|---|---|---|---|---|---|---|---|---|
| `lerobot/diffusion_pusht` | diffusion | 50 | 64.0% [52.0%, 78.0%] | 65.4% | 0.924 [0.878, 0.953] | 262.7 | 73.5 | 8 | 9.19 |
| `lerobot/vqbet_pusht` | vqbet | 200 | 54.0% [47.0%, 61.0%] | 57.0% | 0.818 [0.779, 0.853] | 37.5 | 9.2 | 5 | 1.83 |

- `lerobot/diffusion_pusht`, 50 shared eval seeds: this repo 32/50 (64.0%) against the checkpoint's own block 37/50 (74.0%); 27 episodes agree, 9 succeed only here and 14 only there, exact McNemar p = 0.40, so the two runs are not distinguished at this sample size. Mean coverage 0.914 here against 0.913 there (their `max_reward` is rescaled: it is coverage / 0.95). Their published block is 327/500 = 65.4% over 500 episodes, and these shared seeds are a sub-block of it. Source: `results/diffusion_published.jsonl` (`scripts/published_blocks.py`).

- `lerobot/vqbet_pusht`, 200 shared eval seeds: this repo 108/200 (54.0%) against the checkpoint's own block 113/200 (56.5%); 107 episodes agree, 44 succeed only here and 49 only there, exact McNemar p = 0.68, so the two runs are not distinguished at this sample size. Mean coverage 0.818 here against 0.780 there (their `max_reward` is rescaled: it is coverage / 0.95). Their published block is 285/500 = 57.0% over 500 episodes, and these shared seeds are a sub-block of it. Source: `results/vqbet_published.jsonl` (`scripts/published_blocks.py`).
