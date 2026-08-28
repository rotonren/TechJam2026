# CompassCart Optimization Fusion Round 3: Dense Semantic Rescue

## Goal

Fuse the accepted clarification-alias isolation change with the latest `main`
runtime, then keep Dense retrieval only where it adds production value without
allowing it to disturb stronger lexical or structured evidence.

This work stays on local branch `codex/optimization-fusion-round3`. Nothing is
pushed and no PR is opened or updated without explicit approval.

## Starting Evidence

- Latest remote `main` (`9280c8d`) reports a final public TechnicalScore of
  `0.660411`; no remote ref currently contains the teammate's reported
  near-`0.7` candidate.
- The Round 2 clarification change cherry-picked cleanly onto latest `main`.
- Dense development folds reproduce at mean `0.677997`, standard deviation
  `0.039266`, and selection score `0.658364`.
- With the same code and folds but Dense disabled, lexical/structured retrieval
  reaches mean `0.714376`, standard deviation `0.032101`, and selection score
  `0.698326`.
- Lexical/structured retrieval wins all four folds. Across 160 sessions it has
  five exclusive hits versus three for Dense; among 136 sessions hit by both,
  it has the better terminal rank in 57 versus 32 for Dense (47 ties).

## Hypothesis

The bundled Dense model is useful as a recall mechanism for queries whose
tokens and structured constraints produce no candidates, but its current RRF
weight is too strong when lexical, attribute, or profile evidence already
exists. A deterministic evidence gate can therefore:

1. preserve lexical/structured ordering whenever a positively weighted
   non-Dense source has candidates;
2. invoke Dense only when all positively weighted non-Dense sources are empty;
3. retain semantic recall for unseen terms and paraphrases instead of removing
   the Dense capability from the production system; and
4. avoid unnecessary ONNX inference on ordinary turns.

The gate uses no sample IDs, target IDs, ground truth, catalog-specific values,
or tuned overlap threshold.

## Single Behavioral Variable

- Baseline: run Dense on every turn where the backend is available and the
  component deadline has not expired.
- Candidate: in semantic-rescue mode, run Dense only when every positively
  weighted lexical, attribute, and profile ranking is empty.

All parser behavior, route weights, rank weights, candidate limits, question
policy, response format, and frozen assets remain unchanged.

## Execution

1. Add unit tests for Dense skipping, semantic rescue, disabled-mode backward
   compatibility, configuration validation, and Agent wiring.
2. Implement the gate inside `HybridRetriever`; keep its standalone default
   backward-compatible and enable it through the production `RuntimeConfig`.
3. Run focused retrieval/config/agent tests and Ruff.
4. Run seed `2026`, folds 1-4 CV and compare aggregate, fold, scenario, session,
   fallback, and latency evidence.
5. Run the broader source-controlled suite. Tests that require ignored
   `var/balanced-hardening/` history are reported separately and never satisfied
   with fabricated artifacts.

## Acceptance Gates

1. Mean TechnicalScore is at least `0.704376` (no more than `0.010` below the
   lexical diagnostic).
2. Selection score is at least `0.688326` (same tolerance).
3. Every fold is at least its Dense baseline; no scenario loses a hit versus
   the lexical diagnostic.
4. Dense is demonstrably called when no positively weighted non-Dense evidence
   exists and skipped when such evidence exists.
5. Zero runtime fallback and invalid-response counts; maximum P95 remains below
   `1.5 s`.
6. Focused tests and Ruff pass. Any source-controlled suite failures must be
   shown to be pre-existing artifact or platform portability failures.

If a gate fails, retain the experiment only as local evidence and do not push.

## Result

Accepted for local retention, but not yet PR-ready. The production default now
uses Dense as a semantic rescue only; `HybridRetriever` retains its prior
standalone default for component compatibility.

### Development folds 1-4

| Metric | Dense fusion baseline | Semantic rescue | Delta |
|---|---:|---:|---:|
| Mean TechnicalScore | 0.677997 | 0.714376 | +0.036379 |
| Standard deviation | 0.039266 | 0.032101 | -0.007165 |
| Selection score | 0.658364 | 0.698326 | +0.039962 |

| Fold | Dense fusion | Semantic rescue | Delta | Candidate P50 | Candidate P95 | Candidate max |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.671113 | 0.707295 | +0.036182 | 214.639 ms | 365.460 ms | 581.377 ms |
| 2 | 0.739194 | 0.768714 | +0.029520 | 185.960 ms | 362.291 ms | 581.472 ms |
| 3 | 0.629631 | 0.689604 | +0.059973 | 193.580 ms | 315.808 ms | 419.400 ms |
| 4 | 0.672048 | 0.691893 | +0.019845 | 161.573 ms | 287.184 ms | 396.594 ms |

The candidate wins all four folds, has zero session differences from the
Dense-disabled diagnostic, and has zero runtime fallbacks. Its P50 is lower
than the Dense baseline in every fold. P95 is lower in folds 3 and 4 and higher
in folds 1 and 2, but every P95 and maximum remains below the `1.5 s` gate.

Across pooled scenarios, the candidate matches the Dense-disabled diagnostic:

| Scenario | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Boundary | 0.875000 | 0.487103 | 5.750000 |
| Browsing | 0.890625 | 0.362333 | 3.937500 |
| Buying | 0.937500 | 0.541375 | 3.187500 |
| Intent Override | 0.708333 | 0.488889 | 7.083333 |

The fallback contract is separately covered: Dense is called when all
positively weighted non-Dense rankings are empty, skipped when grounded
evidence exists, and still called when an existing non-Dense ranking has zero
configured weight.

### Verification

- Focused retrieval/config/scenario/fallback suite: `85 passed`.
- Ruff on all changed Python files: passed.
- Full suite: `896 passed`, with the same 13 repository/environment failures
  observed before this change. Twelve read historical files under ignored
  `var/balanced-hardening/`; one compares Windows' `\\?\` symlink target form
  without path normalization. No historical evidence was fabricated.
- `git diff --check`: passed apart from informational Windows line-ending
  conversion warnings.

### Inconclusive full-run attempt

One post-gate 200-session `tools.run_agent` attempt was stopped after nearly 50
minutes because it produced no report or partial output and greatly exceeded
both the fold runtime and the project's resource expectations. Read-only
process checks showed continued CPU work and no network wait, so the command
was not retried. No score is claimed from this attempt.

Before any PR, rerun the full evaluation once in a quiet environment and use
the existing balanced-hardening benchmark transcript to distinguish model
startup cost from per-turn rescue calls. The required ignored baseline and
proxy artifacts are not present in this clone, so proxy/stress/resource gates
remain pending. Nothing from this round has been pushed.
