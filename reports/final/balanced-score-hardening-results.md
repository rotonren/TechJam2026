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
with Dense available and zero fallback in every trial. P0 median trial-P95 was
186.750 ms versus R0 438.090 ms; median initialization was 13479.594 ms
versus 13676.364 ms; median peak memory was 556.453 MiB versus 555.906 MiB.
P0 maximum response time was 292.233 ms.

## Audit Baseline

Previously recorded aggregate-only audit baseline: 394 samples, HitRate@10
0.637056, MRR 0.282756, MTTC 6.139594, efficiency 0.486041, recommended
technical score 0.500563, fallback 0, invalid responses 0, and reported token
usage 0.

Final audit has not run. Public evaluator and public scoring have not run.

## Rejected Stages

### S1 Clarification Alias Gating

S1 was rejected on representative development folds before any stress,
resource, audit, or public run. Mean TechnicalScore changed from `0.548439` to
`0.543191` (`-0.005248`) and selection score changed from `0.543143` to
`0.538259` (`-0.004884`). Boundary lost two hits and Buying HitRate@10 changed
by `-0.021841`; Browsing and Intent Override changed by `+0.001558` and
`+0.012448`. Fallback and invalid-response counts remained zero. The gate
failure codes were `development_selection_regression`,
`development_mean_regression`, `development_buying_regression`, and
`development_boundary_regression`. The S1 production changes were reverted
and are not parents of later experiments.

### S2 Substantive Query History

S2 was rejected on representative development folds against the unchanged P0
parent before any stress, resource, audit, or public run. Mean TechnicalScore
changed from `0.548439` to `0.530068` (`-0.018371`) and selection score changed
from `0.543143` to `0.522261` (`-0.020882`). Boundary lost six hits; Buying,
Browsing, and Intent Override HitRate@10 changed by `-0.035882`, `-0.048286`,
and `-0.029046`. Fold deltas were `-0.013158`, `-0.009796`, `-0.023890`, and
`-0.026642`. Fallback and invalid-response counts remained zero. The gate
failure codes were `development_selection_regression`,
`development_mean_regression`, `development_fold_regression`,
`development_buying_regression`, `development_browsing_regression`,
`development_intent_override_regression`, and
`development_boundary_regression`. The scored runtime fingerprint was
`5736a2fb74069f3ea5639b45630c1ac3e33fca0164e51a8cc78de32df3d5ee00`.
The S2 production changes were reverted and are not parents of later
experiments.

### S3 Attribute Weight

Both predeclared attribute-weight values were rejected on representative
development folds against P0 before any stress, resource, audit, or public
run. At `0.05`, mean TechnicalScore changed by `-0.014804`, selection by
`-0.015492`, and Boundary lost two hits; the gate also rejected Buying,
Intent Override, folds, and mean. At `0.10`, mean changed by `-0.042808`,
selection by `-0.044305`, and Boundary again lost two hits; Buying, Browsing,
Intent Override, every fold, and mean all regressed. Both candidates had zero
fallback and invalid responses. The attribute weight was restored to `0.0`.

### S3 Consensus Bonus

Both predeclared consensus-bonus values were rejected on representative
development folds against P0 before any stress, resource, audit, or public
run. At `0.025`, mean TechnicalScore changed by `-0.004240` and selection by
`-0.004726`; Boundary was unchanged and all scenario declines remained within
their individual limits, but both aggregate gates failed. At `0.05`, mean
changed by `-0.012049`, selection by `-0.013219`, and Browsing HitRate@10 by
`-0.020249`; Boundary was unchanged. Both candidates had zero fallback and
invalid responses. The consensus bonus was restored to `0.0`.
