# CompassCart Score Optimization Results

Recorded on 2026-08-26. The final runtime commit measured here is
`54b2a626878a81c997f5299b39193f9741a120f7`. All selection runs used seed
`2026`, folds 1-4 only, the unchanged evaluator, and the frozen catalog and
dense assets. Fold 5 was not used or inspected during this optimization.

## Selection Evidence

| Candidate | Mean TechnicalScore | Std | Selection score | Decision |
|---|---:|---:|---:|---|
| S0 Windows baseline | 0.519195 | 0.036878 | 0.500756 | Baseline |
| S1 parser and route fixes | 0.560760 | 0.036331 | 0.542594 | Accepted |
| S2 shared semantics | 0.655884 | 0.047840 | 0.631964 | Accepted |
| S3 fusion weight 0.10 | 0.662377 | 0.036433 | 0.644160 | Accepted |
| S4 MMR lambda 1.0 | 0.668189 | 0.043994 | 0.646193 | Rejected |
| Final, MMR lambda 0.85 plus compact cache | 0.662377 | 0.036433 | 0.644160 | Selected |

The final selection score improved by `0.143404` over S0, and the mean
TechnicalScore improved by `0.143182`. Both exceed the required minimums of
`0.510756` selection and `0.529195` mean.

Final fold report: `var/score-optimization/cv/cv-20260826T114942Z.json`.

| Fold | S0 score | Final score | Delta | HitRate@10 | MRR | MTTC | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.479726 | 0.670738 | +0.191012 | 0.850000 | 0.405794 | 4.800 | 398.222 ms |
| 2 | 0.568101 | 0.705902 | +0.137801 | 0.900000 | 0.369673 | 3.750 | 402.422 ms |
| 3 | 0.541363 | 0.604819 | +0.063456 | 0.800000 | 0.284395 | 5.025 | 483.890 ms |
| 4 | 0.487589 | 0.668048 | +0.180459 | 0.850000 | 0.380159 | 4.550 | 379.584 ms |

No fold regressed, all four fold fallback counts were zero, and the highest
fold P95 was `483.890 ms`, below the `1.5 s` limit. Pooled final scenario
HitRate@10 was `0.875000` Boundary, `0.859375` Browsing, `0.890625` Buying,
and `0.708333` Intent Override. Browsing, Buying, and Intent Override improved
over their S0 rates by `0.078125`, `0.140625`, and `0.083333` respectively.

## Official Public Evaluation

The optimization run originally recorded `0.660411`. A release-alignment
reproduction on the same runtime commit with Python `3.12.13`, NumPy `2.5.2`,
ONNX Runtime `1.29.0`, and tokenizers `0.23.1` scored `0.660605` (HitRate@10
`0.840`, MRR `0.376349`, MTTC `4.615`). The `0.000194` score difference is
recorded as compatible dependency-boundary drift; exact input and evaluator
hashes are stored in `final-results.json`. The reproduced value is used by the
owner-review release materials, while the table below preserves the original
experiment output.

The primary comparison is the fresh same-host Windows baseline against the
final candidate. This avoids treating ARM/x86 dense-boundary differences as a
code improvement.

| Metric | Windows baseline | Final | Absolute delta |
|---|---:|---:|---:|
| TechnicalScore | 0.518309 | 0.660411 | +0.142102 |
| HitRate@10 | 0.625000 | 0.840000 | +0.215000 |
| MRR | 0.321365 | 0.376036 | +0.054671 |
| MTTC | 5.530 | 4.620 | -0.910 |
| Efficiency | 0.547000 | 0.638000 | +0.091000 |

TechnicalScore increased by `27.42%` relative to the same-host baseline. The
user-supplied macOS result was `0.525233`; the final Windows score is
`+0.135178` above it, but that cross-platform difference is informational,
not selection evidence.

| Scenario | Baseline Hit | Final Hit | Baseline MRR | Final MRR | Baseline MTTC | Final MTTC |
|---|---:|---:|---:|---:|---:|---:|
| Boundary | 0.600000 | 0.900000 | 0.207778 | 0.392619 | 5.4000 | 5.0000 |
| Browsing | 0.650000 | 0.850000 | 0.282564 | 0.344018 | 5.1625 | 4.6000 |
| Buying | 0.625000 | 0.862500 | 0.366022 | 0.395208 | 5.1750 | 3.7750 |
| Intent Override | 0.566667 | 0.733333 | 0.343611 | 0.404762 | 7.5000 | 6.8000 |

All four scenarios improved HitRate@10, MRR, and MTTC. Boundary gained three
hits with no hit regression.

## Session Changes

The final same-host comparison recovered `53` previously missed sessions and
regressed `10`, for a net recovery of `43`. Baseline misses fell from `75` to
`32`. Among sessions that hit in both versions, `19` improved rank or first-hit
turn and `64` worsened one of those measures.

| Scenario | Recovered | Regressed | Net hits | Rank/turn improved | Rank/turn worsened |
|---|---:|---:|---:|---:|---:|
| Boundary | 3 | 0 | +3 | 1 | 4 |
| Browsing | 22 | 6 | +16 | 8 | 26 |
| Buying | 22 | 3 | +19 | 7 | 28 |
| Intent Override | 6 | 1 | +5 | 3 | 6 |
| Total | 53 | 10 | +43 | 19 | 64 |

The `32` final misses were replayed with the evaluator's exact dialogue and
override schedule, with dense enabled and instrumentation around the four
normal capped candidate sources, fusion, ranking, and response emission.
Terminal classifications are mutually exclusive.

| Scenario | Hard conflict / relaxed | Ranked below Top 10 | Absent from capped sources | Other | Total |
|---|---:|---:|---:|---:|---:|
| Boundary | 0 | 1 | 0 | 0 | 1 |
| Browsing | 3 | 9 | 0 | 0 | 12 |
| Buying | 1 | 8 | 2 | 0 | 11 |
| Intent Override | 1 | 7 | 0 | 0 | 8 |
| Total | 5 | 25 | 2 | 0 | 32 |

Across the full session rather than only the terminal turn, `31/32` targets
entered the ranked pipeline but remained below response Top 10 at every
score-eligible opportunity. Best-rank bins were: `15` at ranks 11-20, `8` at
21-50, `4` at 51-100, `4` at 101+, and `1` never sourced. Non-exclusive source
reach was attribute `30`, lexical `24`, dense `10`, and profile `1`.

Seven targets reached pre-ranker RRF Top 10 on some turn and were then demoted
below response Top 10. At each reachable target's best final-rank turn, the
ranker improved `14` and worsened `17`, so bypassing it wholesale is not safe.

The five terminal hard conflicts were caused by clarification semantics. Two
text-backed style answers became unsupported structured style filters. Three
answers leaked unrelated dynamic aliases into category or size constraints.
Both terminal source-absent cases were Buying: one had appeared earlier before
query-history drift, while the only all-turn capped-source miss had uncapped
turn-one ranks of attribute `206`, lexical `452`, and dense `823` against the
runtime per-source limit of `150`.

## Accepted And Rejected Experiments

- S1 fixed category-span alias pollution, preserved the active route after
  no-preference answers, and kept unknown clarification text as soft evidence.
  Mean/selection changed from `0.519195/0.500756` to
  `0.560760/0.542594`, so it was accepted.
- S2 unified category and open-text semantics across filtering, recall,
  fallback, and ranking. Mean/selection reached `0.655884/0.631964`, a
  selection gain of `0.089370` over S1, so it was accepted.
- S3 restored a bounded normalized fusion contribution at weight `0.10`.
  Mean/selection reached `0.662377/0.644160`, a selection gain of `0.012196`
  over S2 with lower variance, so it was accepted.
- S4 tested only MMR lambda `1.0`. Its mean was `0.668189`, but selection
  improved by only `0.002033`, below the required `0.003`, while standard
  deviation rose from `0.036433` to `0.043994`. It was rejected and the final
  lambda remains `0.85`.
- The exact term-ID searchable cache preserved all S3 scores while reducing
  the single-Agent peak working set below the documented baseline peak.

## Remaining Opportunities

1. Calibrate final ranking, especially bounded normalized attribute and fusion
   evidence. This is the highest-value direction because `25/32` terminal
   misses were already ranked below Top 10, `15` came within ranks 11-20, and
   seven were pre-ranker Top 10. Retain the existing fold and regression gates
   because the current ranker also improves many targets.
2. Tighten clarification alias gating. Catalog-derived aliases should normally
   stay within the pending attribute unless an explicit cue supports another
   attribute. Text-backed clarification values also need a consistent choice
   between structured hard matching and searchable soft evidence. This
   directly addresses the five hard-conflict terminal misses.
3. Exclude no-preference and other control templates from bounded retrieval
   history while preserving the last substantive evidence. Clear it only on a
   real goal override. One terminal source-absent miss is explained by this
   query-history drift.
4. Only after those changes, test a modest or adaptive attribute-source depth
   above `150`. It directly supports only the one all-turn capped-source miss;
   broad dense expansion is not supported by the observed ranks.
5. Re-run the final package on Apple Silicon. The supplied v2 baseline differed
   by `0.006924` between macOS and Windows, so deterministic ID tie-breaking
   should continue to be enforced anywhere floating scores can tie or nearly
   tie.

These are general semantic and ranking changes. No sample ID, ASIN, public
answer, or ground-truth-dependent rule was added to production code.

## Verification

- Ruff: `All checks passed!` for `src`, `tests`, `tools`, and `agent.py`.
- Full pytest: `410 passed in 13.67s`.
- Fallback, performance, agent contract, and submission contract subset:
  `20 passed in 8.27s`.
- Final folds 1-4: zero fallbacks; maximum P95 `483.890 ms`.
- Full dense Agent smoke: dense available, initialization `14084.011 ms`, one
  response `241.339 ms`, `10` recommendations, working set `467.9 MiB`, peak
  `540.5 MiB`. The peak is below the `662.745 MiB` rejection ceiling and
  `35.8 MiB` below the documented `576.3 MiB` baseline peak.
- Full-catalog no-FTS smoke: initialization `11363.570 ms`, two lexical queries
  `290.729 ms` and `255.710 ms`, `150` results each, peak `362.3 MiB`.
- Submission package: `35` files, `32.33 MiB`, SHA256
  `12c9ed5b1723327359a7a5d1dfa513ca58f3286f5941b86e72776e97cf2275c3`.
- Official runtime usage: zero prompt/completion tokens and zero API cost.
- `git diff --check`: clean.
