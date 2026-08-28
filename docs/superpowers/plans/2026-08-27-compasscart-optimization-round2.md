# CompassCart Optimization Round 2: Clarification Alias Isolation

## Goal

Improve private-set generalization by preventing a short clarification answer
from creating unrelated catalog-derived hard constraints. Keep all retrieval
and ranking weights unchanged.

## Evidence

- The accepted v3 baseline has 32 terminal misses on the public set.
- Five misses end in hard conflicts caused by clarification parsing.
- Two text-backed style answers are converted into unsupported structured
  style filters.
- Three answers leak unrelated catalog-derived aliases into category or size
  constraints.
- Round 1's global attribute weight improved the public TechnicalScore by only
  `0.008157` and reduced Boundary MRR and Intent Override HitRate, so it was
  removed from the PR branch and retained only as a local experiment.
- MMR `lambda=1.0` was already tested in the earlier optimization and rejected
  because the selection gain missed its gate and fold variance increased.

## Hypothesis

When an Agent question establishes a pending attribute, catalog-derived aliases
for other attributes in a long reply should be ignored unless the reply
contains a nearby explicit cue for that other attribute. Concise answers and
fixed semantic values such as colors and materials remain recognizable across
attributes, preserving replies such as `Blue leather, please.` and `blue
shoes`. Catalog-derived style text supplied for a style question remains open
query evidence instead of becoming a structured hard filter.

This isolates the current answer to the conversational slot without weakening
initial-message parsing, override parsing, fixed-value recognition, hard-filter
semantics, candidate recall, or final ranking.

## Single Behavioral Variable

- Baseline: any catalog-derived category, size, feature, or use-case alias that
  passes the ordinary message cue rules may become a hard constraint during a
  clarification.
- Candidate: during a long clarification, a catalog-derived alias outside the
  pending attribute requires a nearby explicit cue naming or clearly
  introducing its own attribute; a catalog-derived style answer is handled as
  open evidence rather than a hard style value.

No sample IDs, product IDs, evaluator answers, ranking weights, source depths,
dense assets, or catalog data may enter production rules.

## Execution

1. Add parser tests that fail on cross-attribute catalog alias leakage and pass
   for fixed values plus explicitly cued cross-attribute information.
2. Implement the smallest parser-only gating rule.
3. Run parser/state/integration tests and Ruff.
4. Run the unchanged seed `2026`, folds 1-4 CV against the fresh Round 1
   baseline artifact.
5. Compare aggregate, fold, scenario, and session-level changes.
6. Run the full public evaluator only after the folds accept the candidate.

## Acceptance Gates

1. Selection score improves by at least `0.006` over the fresh baseline
   `0.644411`; this is intentionally stricter than the prior round.
2. Mean TechnicalScore improves by at least `0.006` over `0.662619`.
3. No fold regresses by more than `0.005`.
4. Buying, Browsing, and Intent Override pooled HitRate@10 do not regress.
5. Boundary loses no hit and its pooled MRR does not regress by more than
   `0.01`.
6. Recovered sessions outnumber regressed sessions by at least two.
7. All folds have zero runtime fallbacks and maximum P95 latency remains below
   `1.5 s`.
8. Full pytest and Ruff pass.

If any gate fails, revert the runtime change and retain only the rejected
experiment evidence on the local optimization branch. Nothing from this round
is pushed to the existing PR until the user explicitly approves it.

## Result

Accepted for local retention. The candidate changed only clarification alias
handling in `parser.py`; retrieval depth, ranking weights, dense assets,
catalog, evaluator, and response contract stayed unchanged.

### Development folds 1-4

The same-environment baseline is the fresh Round 1 baseline artifact at runtime
commit `cd51185`. The candidate used the same seed `2026` and fold assignment.

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Mean TechnicalScore | 0.662619 | 0.677997 | +0.015378 |
| Standard deviation | 0.036416 | 0.039266 | +0.002850 |
| Selection score | 0.644411 | 0.658364 | +0.013953 |

| Fold | Baseline | Candidate | Delta | Candidate P95 |
|---|---:|---:|---:|---:|
| 1 | 0.671113 | 0.671113 | 0.000000 | 755.828 ms |
| 2 | 0.706184 | 0.739194 | +0.033010 | 766.039 ms |
| 3 | 0.605131 | 0.629631 | +0.024500 | 891.623 ms |
| 4 | 0.668048 | 0.672048 | +0.004000 | 716.694 ms |

No fold regressed, all four folds had zero runtime fallbacks, and maximum P95
latency remained below `1.5 s`. The candidate recovered the three diagnosed
development sessions (`public_0024`, `public_0073`, and `public_0138`) and
regressed none. Boundary and Intent Override development metrics were
unchanged; Browsing gained two hits and Buying gained one.

### Full 200-session evaluation

The correct project runner is `python -m tools.run_agent`. The unchanged
official evaluator CLI intentionally instantiates `starter.agent.Agent`; a
diagnostic invocation reproduced its documented `0.106710` starter score and
was discarded before the project Agent was run.

| Metric | Stable dense baseline | Round 1 | Round 2 |
|---|---:|---:|---:|
| TechnicalScore | 0.660605 | 0.668762 | 0.681157 |
| HitRate@10 | 0.840000 | 0.845000 | 0.865000 |
| MRR | 0.376349 | 0.386875 | 0.392855 |
| MTTC | 4.615000 | 4.490000 | 4.460000 |

Round 2 improved TechnicalScore by `0.020552` over the stable dense baseline
and by `0.012395` over Round 1. Against the stable dense session report it
recovered five misses, regressed zero hits, improved two sessions already hit,
and worsened zero sessions already hit.

| Scenario | Baseline Hit | Candidate Hit | Baseline MRR | Candidate MRR |
|---|---:|---:|---:|---:|
| Boundary | 0.900000 | 0.900000 | 0.392619 | 0.392619 |
| Browsing | 0.850000 | 0.887500 | 0.345878 | 0.355476 |
| Buying | 0.862500 | 0.875000 | 0.394132 | 0.413299 |
| Intent Override | 0.733333 | 0.766667 | 0.404762 | 0.438095 |

Boundary remained exactly unchanged. Browsing, Buying, and Intent Override all
improved HitRate and MRR, so the candidate avoids Round 1's scenario tradeoff.

### Verification and decision

- Focused parser/state/scenario suite: `69 passed`.
- Full pytest: `424 passed`.
- Ruff: passed for `src`, `tests`, `tools`, and `agent.py`.
- `git diff --check`: clean apart from informational Windows line-ending
  conversion warnings.
- Existing PR after Round 1 rollback: Windows and Ubuntu checks passed, ready
  to merge, no conflicts, owner review still requested.

The implementation is retained only on local branch
`codex/optimization-round2`. It has not been pushed to or described on PR #2.
