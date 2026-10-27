| Question | Measured answer | Source |
|---|---|---|
| Does the harness reproduce the published policy? | Yes: 64.0% [52.0%, 78.0%] success and 0.924 [0.878, 0.953] mean coverage over 50 paired episodes at DDIM N=10, against the model card's 65.4 % / 0.955 | E0 |
| How few sampler steps still work? | 0.121 at N=1; 0.716 at N=2; 0.885 at N=4; 0.919 at N=8; 0.888 at N=10; 0.892 at N=16. The ladder is flat after N=4 (coverage intervals overlap there); chunk latency keeps growing linearly in N, in the table below | E3 |
| What does one chunk cost at batch 1? | 73.5 ms eager / 66.8 ms graph at N=4; 190.5 ms eager / 181.0 ms graph at N=10. CUDA graphs are worth 1.05x here, against 4.7x on the 11 M ablation model. `torch.compile` is NOT RUN (no MSVC on this host) | E9 |
| Does injected observation latency hurt? | success 57% at 0 ms injected vs 43% at 200 ms injected (30 episodes per row, CIs overlapping). The binding constraint is the policy's own cost: 29 deadline misses per episode against a 100 ms period | E10 |
| Does replanning four times faster break it? | Yes for naive replacement: 10% at 100 ms and 10% at 200 ms against 57% / 43% when replanning every 8 steps; uniform temporal ensembling over the same overlapping chunks recovers 53% / 43% at 4x the chunk count | E12 |
