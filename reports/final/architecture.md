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
   lists, applies the shared hard-constraint matcher to every source, and
   combines exact candidates with weighted reciprocal-rank fusion.
4. `ConstraintRanker` keeps exact candidates ahead of explicitly marked
   relaxed alternatives, then diversifies the final ten products.
5. `QuestionPolicy` estimates conversion gain for answerable attributes.
6. `ResponseBuilder` emits unique catalog-valid IDs and zero token usage.

## Judging Map

| Judging concern | Evidence |
| --- | --- |
| Retrieval quality | Field-weighted FTS5, structured attributes, local semantic embeddings, weighted RRF |
| Multi-turn reasoning | Versioned constraint ledger, pending-question tracking, bounded raw query history |
| Intent Override | Goal-level overrides clear obsolete text/questions; attribute-level corrections replace only the named slot |
| Browsing utility | Profile-aware broad recall and conversion-gain clarification |
| Reliability | Shared hard-filter matcher, disclosed relaxation evidence, bounded state/traces, component fallbacks, valid-ID filtering |
| Feasibility | CPU ONNX int8 assets, 45 MiB runtime asset set, zero API cost, offline operation; full-catalog dense smoke was 504 MiB working set after one response |
| Reproducibility | Deterministic CV, tagged candidate and sealed audit fold, checksummed assets, deterministic allowlist ZIP |

## State and Override Semantics

Each constraint records source, confidence, turn, intent version, hardness,
operator, and status. Active constraints belong to the current intent, except a
compatible category and soft profile preferences may be retained. A goal-level
override starts a new version, supersedes old user constraints, clears old
free-text evidence and question state, and prevents the message from being
misread as an answer to the previous clarification. An attribute-level
correction replaces only the named slot. A no-preference reply is stored
explicitly so the agent does not ask the same boundary question again.

## Failure Containment

Dense load and inference failures return an unavailable backend. FTS errors use
the Python catalog path. Parser, router, retrieval, ranking, question, and trace
failures are isolated at the orchestrator boundary. Recommendations are always
deduplicated against the frozen catalog, with popular items as the last resort.
The runtime never sends network requests or reads credentials.

## Evaluation Discipline

The 200 public sessions are deterministically stratified by scenario and
difficulty. The optimized runtime at commit `54b2a62` was selected only from
folds 1-4 using `mean(TechnicalScore) - 0.5 * std(TechnicalScore)` (`0.644160`).
Fold 5 was viewed during the historical v2 release, so it is no longer described
as a blind audit for this candidate. Final public evaluation is reported only
after selection; the organizer's private 800-session set remains the true blind
test. The official evaluator, catalog, public labels, and dense assets are never
modified, and their release fingerprints are recorded in `final-results.json`.
