# CompassCart Final Stable Score

## Result

Runtime candidate: `c0d444fa6f53b8b91d8b268f0f7273737eb78cd0`.

| Metric | Final stable | Previous stable | Change |
| --- | ---: | ---: | ---: |
| TechnicalScore | 0.660411 | 0.660411 | 0.000000 |
| HitRate@10 | 0.840000 | 0.840000 | 0.000000 |
| MRR | 0.376036 | 0.376036 | 0.000000 |
| MTTC | 4.620 | 4.620 | 0.000 |
| Efficiency | 0.638000 | 0.638000 | 0.000000 |

The final public result is exactly identical to the previous stable result at
`b641ff97b0f4d7ae0c2fc7646e250492370231bd`. The hardening work therefore
improved release safety and resource evidence without claiming a score gain.

## Scenario Aggregates

| Scenario | Samples | HitRate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Boundary | 10 | 0.900000 | 0.392619 | 5.000 |
| Browsing | 80 | 0.850000 | 0.344018 | 4.600 |
| Buying | 80 | 0.862500 | 0.395208 | 3.775 |
| Intent Override | 30 | 0.733333 | 0.404762 | 6.800 |

The fail-closed public wrapper recorded 892 attempted responses and 892 traces,
with consistent call counts and zero fallback.

## Independent Gates

- The one sealed audit passed with TechnicalScore `0.500563`, HitRate@10
  `0.637056`, MRR `0.282756`, MTTC `6.139594`, zero fallback, and zero invalid
  responses.
- The three-trial resource benchmark passed with Dense available and zero
  fallback. P95 was `183.692 ms`, maximum latency was `529.531 ms`,
  initialization was `13219.807 ms`, and peak working set was `557.008 MiB`.
- P95 improved `58.070%` against the compatible R0 benchmark.
- The full automated suite passed `891` tests with `7` skipped; all `9` frozen
  input checks and all `51` delivery-contract checks passed. Ruff lint passed.
- macOS verification is pending.

## Selection Decision

S1, S2, every S3 factor, and S4 were rejected by their precommitted
development gates and fully reverted. S5 was deferred to finish the stable
delivery. No rejected scoring experiment is present in the runtime candidate.

This report and its paired JSON contain aggregate evidence only. The protected
raw evaluation output is intentionally excluded from the source bundle.
