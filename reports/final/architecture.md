# CompassCart Architecture and Judging Evidence

## Design Goal

CompassCart optimizes early exact-product discovery while remaining reliable on
an offline CPU host. Advanced retrieval is optional by construction: the same
Agent contract stays valid when ONNX Runtime, FTS5, model assets, or tracing are
unavailable.

## Request Flow

1. `SessionStore` parses the new turn into versioned constraints and bounded
   query evidence.
2. `RoutePlanner` selects Buying, Browsing, or Intent Override weights.
3. `HybridRetriever` generates lexical, attribute, profile, and dense candidate
   lists and combines them with weighted reciprocal-rank fusion.
4. `ConstraintRanker` enforces active constraints, penalizes conflicts, and
   diversifies the final ten products.
5. `QuestionPolicy` estimates conversion gain for answerable attributes.
6. `ResponseBuilder` emits unique catalog-valid IDs and zero token usage.

## Judging Map

| Judging concern | Evidence |
| --- | --- |
| Retrieval quality | Field-weighted FTS5, structured attributes, local semantic embeddings, weighted RRF |
| Multi-turn reasoning | Versioned constraint ledger, pending-question tracking, bounded query history |
| Intent Override | Override-first parsing clears obsolete text and supersedes conflicting constraints |
| Browsing utility | Profile-aware broad recall and conversion-gain clarification |
| Reliability | Contract sanitizer, bounded state/traces, component fallbacks, valid-ID filtering |
| Feasibility | CPU ONNX int8 assets, 45 MiB runtime asset set, zero API cost, offline operation |
| Reproducibility | Deterministic CV, sealed audit fold, checksummed assets, deterministic allowlist ZIP |

## State and Override Semantics

Each constraint records source, confidence, turn, intent version, hardness, and
status. Active constraints belong to the current intent, except a compatible
category may be retained. An override starts a new version, supersedes
conflicts, clears old free-text evidence, and prevents the message from being
misread as an answer to the previous clarification. A no-preference reply is
stored explicitly so the agent does not ask the same boundary question again.

## Failure Containment

Dense load and inference failures return an unavailable backend. FTS errors use
the Python catalog path. Parser, router, retrieval, ranking, question, and trace
failures are isolated at the orchestrator boundary. Recommendations are always
deduplicated against the frozen catalog, with popular items as the last resort.
The runtime never sends network requests or reads credentials.

## Evaluation Discipline

The 200 public sessions are deterministically stratified by scenario and
difficulty into five folds. Folds 1-4 select one configuration using
`mean(TechnicalScore) - 0.5 * std(TechnicalScore)`. Fold 5 remains sealed until
the selected commit is tagged, then is run once without further tuning. The
official evaluator and catalog are never modified.
