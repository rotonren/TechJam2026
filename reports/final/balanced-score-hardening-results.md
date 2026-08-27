# Balanced Score Hardening Results

Foundation commit: `ac36def`.

## P0 Proxy Evidence

| Suite | Mean technical | Selection | Std technical | Fallback | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development, representative folds 1-4 | 0.548439 | 0.543143 | 0.010594 | 0 | 0 |
| Stress | 0.596585 | 0.596585 | 0.000000 | 0 | 0 |

The P0 development and stress aggregates are unchanged from their respective
foundation baselines. The frozen proxy manifest hash is
`9dfb5694a7b733952c40ea4be4661480ed8d01adb28f8fb0dc73e8f8a89f7c9d`.

## Runtime Evidence

The R0 and P0 release benchmarks used the same frozen transcript hash,
`77f4399d65cabff5fcb5f0006132ba43fc47bf8571737ba6a88465ebb7066590`,
with Dense available and zero fallback in every trial. P0 median P95 was
186.956 ms versus R0 449.975 ms; median initialization was 13479.594 ms
versus 13676.364 ms; median peak memory was 556.453 MiB versus 555.906 MiB.
P0 maximum response time was 292.233 ms.

## Audit Baseline

Previously recorded aggregate-only audit baseline: 394 samples, HitRate@10
0.637056, MRR 0.282756, MTTC 6.139594, efficiency 0.486041, recommended
technical score 0.500563, fallback 0, invalid responses 0, and reported token
usage 0.

Final audit has not run. Public evaluator and public scoring have not run.
