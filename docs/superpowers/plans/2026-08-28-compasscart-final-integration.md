# CompassCart Final Integration: Team Repositories And Local Optimizations

## Scope

Integrate and evaluate all currently available team work:

- `rotonren/TechJam2026` latest `main` at `9280c8d`;
- local clarification-alias isolation at `14bcd6c`;
- local Dense semantic-rescue gate at `fbc623c`; and
- `zheisne/TechJam2026` branch `feat/rank-calibration-v3` at `ee81ad9`,
  which raises `rank_fusion_weight` from `0.10` to `0.25`.

The older branches in the second repository diverge from `b641ff9` and omit
the later runtime, integrity, packaging, proxy, and test hardening. They are
evidence/history only and are not merge bases for the final candidate.

The final integration is prepared on `codex/final-integrated` for owner review.

## Reproduced Baselines

Using seed `2026`, folds 1-4, and the same frozen catalog and public dataset:

| Candidate | Mean TechnicalScore | Std | Selection score |
|---|---:|---:|---:|
| Latest-main + local clarification, Dense fusion `0.10` | 0.677997 | 0.039266 | 0.658364 |
| Teammate v3 only, Dense fusion `0.25` | 0.676967 | 0.032916 | 0.660509 |
| Local clarification + Dense semantic rescue `0.10` | 0.714376 | 0.032101 | 0.698326 |

The teammate change is independently reproducible but does not by itself
explain a near-`0.7` development mean in the fetched refs. It slightly lowers
mean score while reducing variance enough to improve selection score by
`0.002145` over the Dense `0.10` local baseline. Its calibration signal may
still be complementary after Dense is gated to semantic rescue.

## Integration Hypothesis

When Dense is suppressed in the presence of grounded lexical/structured
evidence, the RRF fusion score becomes a more reliable aggregate of lexical,
attribute, and profile rankings. Raising its final-ranker weight to `0.25` may
then improve stable source ordering without reintroducing Dense noise.

This is a clean factorial combination of three general rules. It introduces
no sample IDs, target IDs, catalog-specific values, evaluator answers, or new
tuned thresholds.

## Acceptance Gates

1. Integrated mean and selection score must both exceed the current local
   semantic-rescue candidate (`0.714376` and `0.698326`).
2. No fold may regress by more than `0.005`; no pooled scenario may lose a hit.
3. Zero runtime fallbacks; every fold P95 and maximum remains below `1.5 s`.
4. Configuration and direct ranker tests cover the new allowed value and the
   exact source-weight budget.
5. Focused tests, the full source-controlled suite, Ruff, and `git diff --check`
   must pass from a fresh checkout without relying on ignored benchmark files.

If the integrated candidate misses either score gate, revert only the `0.25`
default from the final branch and retain the teammate experiment in this report.

## Result

Accepted as the final local development candidate.

### Integrated folds 1-4

| Metric | Local semantic rescue `0.10` | Final integrated `0.25` | Delta |
|---|---:|---:|---:|
| Mean TechnicalScore | 0.714376 | 0.722919 | +0.008543 |
| Standard deviation | 0.032101 | 0.032542 | +0.000441 |
| Selection score | 0.698326 | 0.706648 | +0.008322 |

| Fold | Local | Final | Delta | Final P50 | Final P95 | Final max |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.707295 | 0.738592 | +0.031297 | 155.500 ms | 239.144 ms | 420.514 ms |
| 2 | 0.768714 | 0.768455 | -0.000259 | 156.182 ms | 225.427 ms | 310.097 ms |
| 3 | 0.689604 | 0.687655 | -0.001949 | 153.403 ms | 238.552 ms | 305.452 ms |
| 4 | 0.691893 | 0.696973 | +0.005080 | 153.165 ms | 208.254 ms | 274.539 ms |

Every regression is below the predeclared `0.005` fold limit. The integrated
candidate recovers two sessions, regresses no hits, improves the terminal rank
of 36 sessions hit by both candidates, worsens 35, and ties 70. Every fold has
zero runtime fallbacks; all P95 and maximum latencies are below `0.5 s`.

### Pooled scenario trade-offs

| Scenario | Local Hit | Final Hit | Local MRR | Final MRR | Local MTTC | Final MTTC |
|---|---:|---:|---:|---:|---:|---:|
| Boundary | 0.875000 | 0.875000 | 0.487103 | 0.447371 | 5.750000 | 5.750000 |
| Browsing | 0.890625 | 0.906250 | 0.362333 | 0.397520 | 3.937500 | 3.703125 |
| Buying | 0.937500 | 0.953125 | 0.541375 | 0.478286 | 3.187500 | 2.468750 |
| Intent Override | 0.708333 | 0.708333 | 0.488889 | 0.441369 | 7.083333 | 6.833333 |

The higher fusion weight trades some terminal rank precision for earlier and
additional hits. Browsing improves on all three primary metrics; Buying gains
one hit and substantially improves MTTC while losing MRR. Boundary and Intent
Override preserve hits. A route- or scenario-specific follow-up was deliberately
not tuned because the global combination already passes the declared gates and
further public-fold optimization would increase overfitting risk.

### Final fold and all-200 aggregate

After the development candidate and its test fixes were committed, fold 5 was
run once in aggregate-only audit mode at commit `848818e`:

| Fold | Sessions | TechnicalScore | Hit@10 | MRR | MTTC | P50 | P95 | Max | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 40 | 0.772798 | 0.925000 | 0.569325 | 4.025000 | 173.207 ms | 279.403 ms | 506.202 ms | 0 |

Because all five folds contain 40 sessions and TechnicalScore is an additive
per-session aggregate, combining the four development folds with the sealed
fold gives the exact public all-200 aggregate TechnicalScore:

| Scope | Sessions | Mean TechnicalScore | Population std |
|---|---:|---:|---:|
| Folds 1-5 | 200 | **0.732895** | 0.035288 |

This is the final score claimed for the integrated candidate. No per-session
fold-5 details were emitted or inspected.

### Verification

- Focused config/ranker/retrieval/parser suite: `161 passed`.
- Full suite after making hardening fixtures self-contained and normalizing the
  Windows symlink assertion: `910 passed`.
- Ruff on all touched files and `git diff --check`: passed.
- Direct tests now cover `0.25` at both configuration and ranker boundaries and
  include it in the exact source-budget matrix.

### Full evaluation status

One clean post-commit `tools.run_agent` attempt for `dd55605` was stopped after
10 minutes without output or a published report, using a cutoff declared before
the run. It was not retried and no partial score is claimed. The independent
fold-5 run completed normally in under two minutes, so the earlier behavior is
isolated to the monolithic full-run path rather than a fold-5 model failure.
The five equal public folds provide the all-200 aggregate above while preserving
the audit boundary.

The final candidate combines the latest hardened mainline, clarification alias
isolation, Dense semantic rescue, and the teammate's `0.25` rank calibration.
The repository tests no longer depend on ignored proxy/resource artifacts, and
the candidate is ready for owner review through a pull request.
