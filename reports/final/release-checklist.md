# CompassCart Release Checklist

## Candidate Integrity

- [x] Runtime configuration selected only from stratified development folds 1-4.
- [x] Candidate tagged `compasscart-audit-candidate` before fold 5 was viewed.
- [x] Sealed fold 5 executed exactly once; no post-audit tuning performed.
- [x] Official evaluator, public labels, contract, and catalog were not modified.
- [x] Final evidence records candidate commit and configuration hash.

## Runtime and Reliability

- [x] Agent exports the required `reset` and `respond` methods.
- [x] Recommendations are unique, catalog-valid, ordered, and limited to ten.
- [x] Component failures degrade to deterministic valid output.
- [x] 800-session bounded-state and latency regression test passes.
- [x] Network and credentials are unnecessary; runtime token usage and API cost are zero.
- [x] `COMPASSCART_DISABLE_DENSE=1` lexical-only matrix passes.

## Assets and Package

- [x] Five required dense assets match `assets/SHA256SUMS`.
- [x] The unquantized 90 MB intermediate model is excluded.
- [x] Package is generated from a deterministic allowlist and secret-scanned.
- [x] Public labels, evaluator, organizer files, caches, traces, and experiments are excluded.
- [x] Extracted package imports `Agent` and produces a fixture response.
- [x] Clean virtual-environment install and offline smoke command are verified.
- [x] Data and model attribution notices are included.
- [x] Final ZIP size and SHA256 are printed by `python -m tools.package_submission`.

## Submission Materials

- [x] README contains Python version, install, run, fallback, metrics, costs, and limitations.
- [x] Architecture report maps implementation evidence to judging concerns.
- [x] Devpost draft contains problem, innovation, results, feasibility, and team roles.
- [x] Three-minute demo covers Browsing, Intent Override, and network-disabled operation.
- [ ] Replace five role placeholders with final member names before portal submission.
- [ ] Record the final ZIP SHA256 in the submission portal and team handoff sheet.
- [ ] Capture the demo video and verify audio, terminal readability, and time limit.
