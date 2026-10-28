| Dtype | Mode | N | Mean chunk ms | p95 ms | Control Hz | Peak VRAM MiB | Fits 100 ms |
|---|---|---|---|---|---|---|---|
| bfloat16 | cudagraph | 4 | 55.6 | 59.1 | 17.99 | 710 | yes |
| bfloat16 | eager | 4 | 65.6 | 68.9 | 15.24 | 646 | yes |
| float16 | cudagraph | 4 | 56.5 | 57.1 | 17.69 | 774 | yes |
| float16 | eager | 4 | 68.2 | 73.6 | 14.66 | 710 | yes |
| float32 | cudagraph | 4 | 73.4 | 78.5 | 13.63 | 1189 | yes |
| float32 | eager | 4 | 78.1 | 81.0 | 12.81 | 1125 | yes |
| bfloat16 | cudagraph | 10 | 138.4 | 146.0 | 7.23 | 742 | no |
| bfloat16 | eager | 10 | 152.3 | 157.6 | 6.57 | 678 | no |
| float16 | cudagraph | 10 | 136.7 | 138.7 | 7.31 | 806 | no |
| float16 | eager | 10 | 155.3 | 158.3 | 6.44 | 742 | no |
| float32 | cudagraph | 10 | 194.4 | 201.2 | 5.14 | 1221 | no |
| float32 | eager | 10 | 209.0 | 226.0 | 4.79 | 1157 | no |

Same policy, same paired episodes, only the arithmetic dtype changes:

| Dtype | N | Episodes | Success (95% CI) | Mean coverage | Mean chunk ms | Wall s |
|---|---|---|---|---|---|---|
| bfloat16 | 4 | 40 | 50.0% [35.0%, 65.0%] | 0.885 | 62.7 | 104 |
| float16 | 4 | 40 | 45.0% [30.0%, 60.0%] | 0.901 | 62.8 | 106 |
| float32 | 4 | 40 | 35.0% [22.5%, 50.0%] | 0.900 | 79.3 | 131 |
