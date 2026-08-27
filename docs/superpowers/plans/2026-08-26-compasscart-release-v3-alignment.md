# CompassCart Optimized Candidate Release Alignment

## Goal

Turn the already-implemented optimized runtime into an owner-reviewable release
candidate without changing scored Agent behavior.

## Baseline

- Branch base: `main` at `b641ff9`
- Runtime commit: `54b2a62`
- Proposed post-review tag: `compasscart-v3-candidate`
- Development folds 1-4: mean `0.662377`, selection score `0.644160`
- Fresh dense public reproduction: TechnicalScore `0.660605`
- Fresh lexical-only operations control: TechnicalScore `0.704353`

The lexical control is recorded but is not version-selection evidence. The
default dense path remains unchanged to avoid choosing production behavior from
the public set.

## Execution Order

1. Download and verify the official 50,000-product catalog.
2. Reproduce the public evaluator with the dense runtime explicitly available.
3. Record exact dependency and input fingerprints in `final-results.json`.
4. Align README, architecture, Devpost, demo, ablation, and release checklist.
5. Add contract tests that prevent release metadata drift.
6. Include score-optimization evidence in the deterministic submission package.
7. Add a release-audit command that rejects missing or silently disabled dense.
8. Make configuration fingerprints canonical across Windows and Linux.
9. Add GitHub Actions gates for Ruff and the complete fixture-based test suite.
10. Run all tests, lint, release audit, package reproducibility, and Git checks.
11. Push only the review branch; the owner merges and creates the candidate tag.

## Release Integrity Rules

- Fold 5 remains historical v2 evidence and is not called blind for this candidate.
- The organizer's private 800 sessions remain the true blind test.
- Catalog, evaluator, public labels, dense assets, and runtime source fingerprints
  are recorded separately from aggregate metrics.
- `data/catalog.jsonl`, `.venv`, `var`, `dist`, caches, credentials, and
  session-level result files remain untracked.
- The branch must not change `src/compasscart` scored behavior.

## Owner Review Gate

The owner should verify the PR checks, review all claims, fill team names, merge
the branch, create `compasscart-v3-candidate` on the merged commit, rebuild the
deterministic ZIP, and record its final SHA256 in the portal and handoff sheet.
