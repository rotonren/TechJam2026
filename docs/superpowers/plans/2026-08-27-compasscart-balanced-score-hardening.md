# CompassCart Balanced Score Hardening Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task by task. Every behavior change also uses `superpowers:test-driven-development`; every accepted stage receives specification review and then code-quality review before it is committed.

**Goal:** Improve private-set generalization and the final competition score through bounded semantic, ranking, question-policy, and recall changes while preserving the official contract, offline operation, deterministic ordering, and resource limits.

**Architecture:** Keep `agent.Agent` and the current Parser -> SessionStore -> Router -> HybridRetriever -> ConstraintRanker -> QuestionPolicy pipeline. Add only internal evidence and fixed, bounded controls. Select variants on representative development folds 1-4, use stress only as a veto, keep frozen audit fold 5 sealed until one final run, and run the public evaluator exactly once after the candidate is immutable.

**Tech Stack:** CPython 3.12, dataclasses, SQLite FTS5, NumPy, ONNX Runtime, pytest, Ruff, PowerShell, and the existing sealed proxy/benchmark/package tools.

---

## Non-Negotiable Guardrails

Work only in `C:\Users\renha\Documents\ChatGPT\techjam\.worktrees\compasscart` on branch `codex/compasscart`.

Never modify, stage, package, or delete these user-owned untracked files:

- `docs/compasscart-agent-architecture-analysis.docx`
- `reports/final/score-results-b641ff9-2026-08-26.json`
- `reports/final/score-results-b641ff9-2026-08-26.md`

Never modify the evaluator, public set, catalog, Dense assets, or scoring formula. Before each scoring checkpoint and final packaging, verify these frozen SHA-256 values:

| Path | SHA-256 |
| --- | --- |
| `evaluator/local_evaluator.py` | `84EA899707452DE249CA62ABEE77C4B40AB7A3139B5CC798AC30C9F521F91B30` |
| `data/public_set.jsonl` | `571359A8A69014C43FC30D39C996C4A28E875DCCC249DFFC707358757BEB16C0` |
| `data/catalog.jsonl` | `DA979B05A68AF864CB0DCF9EE6A81C010C7E66A57978AD286C7A2E005FC69A67` |
| `assets/SHA256SUMS` | `2857869C2A872CCEA9D93BB043B8CB45EEE07CB1EFC1F943B401C1919982D86E` |
| `assets/model/model.int8.onnx` | `3013F5CDB68EA6B6A271AB8FEF96C5E6721669C2C2BE3F83EC1BE07486133892` |
| `assets/model/tokenizer.json` | `DA0E79933B9ED51798A3AE27893D3C5FA4A201126CEF75586296DF9B4D2C62A0` |
| `assets/product_vectors/product_ids.npy` | `E5AB6608C15DD0B51DD2F63DB088705613EFDFEA85859462C2D514752FE8D7C9` |
| `assets/product_vectors/scales.npy` | `3EB26371CB15A3E2AF5D287A290CD338C12C3A3F9E606BDD911C53E6D4064D53` |
| `assets/product_vectors/vectors.int8.npy` | `CCAF43034103312788DDDE27890861C6F5D93052DBC930B0B1BFF56ACF0C4D63` |

Production code must contain no public or proxy sample IDs, target ASINs, ground truth, failure-specific rules, or audit data. Do not inspect audit sessions. All intermediate proxy and benchmark artifacts stay under ignored `var/balanced-hardening/`.

Foundation parent evidence is fixed:

- development parent: `var/balanced-hardening/proxy-v1/dev-p0.json`
- stress parent: `var/balanced-hardening/proxy-v1/stress-p0.json`
- audit aggregate parent: `var/balanced-hardening/proxy-v1/audit/baseline.json`
- resource parent: `var/balanced-hardening/benchmark-r0-wall-v2.json`
- frozen transcript SHA-256: `77f4399d65cabff5fcb5f0006132ba43fc47bf8571737ba6a88465ebb7066590`

Do not run an audit-labeled command in Tasks 1-7. Do not run the public evaluator in Tasks 1-7.

## Stage Gate Contract

Every accepted S-stage or accepted S3 factor is compared with its immediate accepted parent, never with the best result seen later.

| Gate | Required result |
| --- | --- |
| Development selection | `delta >= +0.003`; S1/S2 correctness fixes may use `delta >= -0.001` |
| Development mean TechnicalScore | no decline |
| Per-fold TechnicalScore | no fold declines by more than `0.015` |
| Buying/Browsing/Intent Override HitRate@10 | each decline no more than `0.02` |
| Development Boundary | candidate hit count cannot be below parent |
| Stress overall TechnicalScore | decline no more than `0.01` |
| Stress major-scenario HitRate@10 | each decline no more than `0.025` |
| Stress Boundary | candidate hit count cannot be below parent |
| Runtime validity | candidate fallback and invalid-response counts both equal zero |
| Resource compatibility | Dense available; fallback zero; identical frozen transcript/catalog/platform provenance |
| Resource regression | R0-relative P95/init/peak medians each at most `+5%`; max response `<1.5 s` |

For a parameter set, run development first. Select the highest development `selection_score` among values that pass all development gates; break an exact tie in favor of the smaller change. Run stress only for that one development-selected value. Stress may reject it but must never choose another value. If rejected, retain the parent and do not try the runner-up. This prevents stress-set parameter tuning.

Rejected experiments do not remain in production code and are recorded, with aggregate reason only, in `reports/final/balanced-score-hardening-results.md`. Restore only files owned by that experiment with `apply_patch`; never use a destructive worktree reset and never touch the three protected untracked files.

## File Map

- Create `tools/compare_proxy_stages.py`: strict, aggregate stage-gate comparator.
- Create `tests/unit/test_compare_proxy_stages.py`: comparator schema and gate tests.
- Create `tools/runtime_fingerprint.py`: deterministic production-source/config fingerprint shared by experiment tools.
- Create `tools/verify_frozen_inputs.py`: fail-closed nine-file integrity preflight.
- Create `tests/unit/test_runtime_fingerprint.py`: stable fingerprint tests.
- Create `tests/unit/test_verify_frozen_inputs.py`: frozen-input verifier tests.
- Modify `tools/run_proxy.py` and `tests/unit/test_run_proxy.py`: bind normal reports to runtime/config fingerprints.
- Modify `tools/benchmark_release.py` and `tests/unit/test_benchmark_release.py`: add resource-only comparison mode and fingerprints.
- Modify `tools/run_agent.py`: final public-run trace/fallback evidence without changing evaluator behavior.
- Create `tests/unit/test_run_agent.py`: public wrapper evidence tests.
- Modify `src/compasscart/parser.py`: S1 clarification gating and S2 substantive evidence signal.
- Modify `src/compasscart/state.py`: S2 filtered query history.
- Modify `src/compasscart/agent.py`: S2 query composition, S4 callbacks, S5 configuration wiring.
- Modify `src/compasscart/models.py`: compatible internal candidate evidence fields.
- Modify `src/compasscart/config.py`: validated bounded ranking and attribute-depth controls.
- Modify `src/compasscart/retrieval.py`: source ranks, pre-rank, and adaptive attribute depth.
- Modify `src/compasscart/ranker.py`: bounded evidence features and adaptive MMR gate.
- Modify `src/compasscart/question_policy.py`: answerability/parser/retrieval-aware utility.
- Modify focused unit, contract, and integration tests alongside their production modules.
- Create `tools/package_source.py` and `tests/unit/test_package_source.py`: deterministic source bundle with an explicit allowlist.
- Create `reports/final/balanced-score-hardening-results.md`: accepted/rejected aggregate evidence and final delivery summary.

Use these shell variables in every PowerShell command block:

```powershell
$env:PYTHONPATH = "src;."
$python = ".\.venv\Scripts\python.exe"
$ruff = ".\.venv\Scripts\ruff.exe"
$verifyFrozen = { & $python -m tools.verify_frozen_inputs }
```

### Task 1: Automated Development and Stress Comparator

**Files:**

- Create: `tools/compare_proxy_stages.py`
- Create: `tests/unit/test_compare_proxy_stages.py`
- Create: `tools/runtime_fingerprint.py`
- Create: `tools/verify_frozen_inputs.py`
- Create: `tests/unit/test_runtime_fingerprint.py`
- Create: `tests/unit/test_verify_frozen_inputs.py`
- Modify: `tools/run_proxy.py`
- Modify: `tests/unit/test_run_proxy.py`
- Modify: `tools/benchmark_release.py`
- Modify: `tests/unit/test_benchmark_release.py`

- [ ] **Step 1: Write failing schema and provenance tests**

Build minimal parent/candidate reports with the real `run_proxy.py` shape. Assert rejection when suite, manifest hash, dataset hash, fold IDs, sample-ID sets, or scenario sets differ. Assert representative development contains exactly folds 1-4 and stress contains its single complete selection. Require parent development/stress to share a config fingerprint and source identity; require candidate development/stress to share both `config_hash` and `runtime_hash`.

```python
def test_comparator_rejects_mismatched_provenance(): ...
def test_comparator_rejects_dev_stress_runtime_or_config_mismatch(): ...
def test_comparator_rejects_missing_or_extra_sessions(): ...
def test_comparator_requires_dev_folds_one_through_four(): ...
```

Run:

```powershell
& $python -m pytest tests/unit/test_compare_proxy_stages.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 2: Write failing tests for every gate**

Test ordinary `+0.003` and correctness `-0.001` selection thresholds separately. Derive correctness tolerance from the controlled stage family: S1/S2 use it; S3/S4/S5 cannot request it. Test mean, each fold, each of Buying/Browsing/Intent Override HitRate independently, Boundary hit counts from sessions, stress overall/scenarios/Boundary, and zero fallback/invalid counts. The report must expose aggregate deltas and failure codes, not sample IDs.

```python
def test_scoring_stage_requires_three_thousandths_selection_gain(): ...
def test_correctness_stage_allows_at_most_one_thousandth_selection_loss(): ...
def test_scoring_stages_cannot_request_correctness_tolerance(): ...
def test_each_major_scenario_is_an_independent_hard_gate(): ...
def test_fold_scenario_and_boundary_regressions_each_reject(): ...
def test_stress_is_a_veto_and_cannot_select_a_parameter(): ...
def test_fallback_or_invalid_response_rejects(): ...
```

- [ ] **Step 3: Implement the comparator and CLI**

Implement pure helpers that validate reports before comparing them:

```python
@dataclass(frozen=True)
class StageGatePolicy:
    minimum_selection_delta: float
    minimum_mean_delta: float = 0.0
    maximum_fold_decline: float = 0.015
    maximum_dev_scenario_decline: float = 0.02
    maximum_stress_decline: float = 0.01
    maximum_stress_scenario_decline: float = 0.025

def compare_development(parent_dev, candidate_dev, *, correctness_fix): ...
def compare_stress(parent_stress, candidate_stress): ...
def compare_stage(parent_dev, candidate_dev, parent_stress, candidate_stress, *, correctness_fix): ...
```

The CLI accepts `--phase development|complete`, `--parent-dev`, `--candidate-dev`, optional stress/resource paths required only for `complete`, `--stage`, and `--output`. The controlled `--stage` family derives the selection threshold; there is no general correctness bypass flag. Development phase applies every development gate without opening or requiring stress/resource results and writes a receipt containing the stage, policy, candidate report SHA-256, runtime/config fingerprints, and accepted status. Complete phase requires that exact development receipt via `--development-receipt`, revalidates its hash and inputs, repeats development validation, verifies candidate dev/stress/resource runtime/config fingerprints are identical, recomputes the current worktree runtime/config fingerprints and requires an exact match, then applies the stress and resource vetoes. It writes canonical JSON, exits zero only when accepted, and never accepts NaN, infinity, booleans as numbers, rounded aggregate inconsistencies, missing sessions, or mismatched provenance.

- [ ] **Step 4: Add runtime fingerprints and a fail-closed integrity preflight**

`tools.runtime_fingerprint` hashes relative path plus bytes for root `agent.py` and every sorted `src/compasscart/*.py`; it also canonicalizes `RuntimeConfig` into the same config hash used by proxy and benchmark tools. Normal proxy reports add `runtime_hash` without changing the exact aggregate-only audit schema. Candidate dev/stress equality is mandatory. The legacy P0 parent may use its matching recorded `commit` plus `config_hash` because its reports predate `runtime_hash`; no later report may omit the new field.

`tools.verify_frozen_inputs` contains exactly the nine path/hash pairs from Guardrails, resolves them below the repository root, streams each file, prints only path/status, and exits nonzero on missing, extra, or mismatched input. Tests use an injected temporary manifest/root; they do not mutate real frozen files.

- [ ] **Step 5: Add a resource-only benchmark comparison mode**

Preserve the existing default equivalent-output mode for P0. Add explicit `--comparison-mode resource` for behavior-changing S stages. Resource mode still requires identical trial count, transcript/catalog/catalog-snapshot/capture/platform/CWD/peak provenance, Dense availability, zero fallback, internally identical candidate response hashes across all three trials, P95/init/peak at most 5% worse than R0, and max below 1.5 s. It intentionally does not require candidate `response_hash == R0 response_hash` and does not require P0's material-gain threshold. Add candidate `runtime_hash` and `config_hash` to the aggregate report and validate them across worker trials.

RED tests cover default-mode response mismatch rejection, resource-mode response mismatch allowance, within-candidate determinism, all provenance/resource limits, and CLI mode validation.

- [ ] **Step 6: Verify and commit**

Before committing, initialize `reports/final/balanced-score-hardening-results.md` with the foundation commit, P0 development/stress aggregates, R0/P0 runtime evidence, audit baseline aggregate, and explicit statements that final audit/public scoring have not run. Do not include session IDs.

```powershell
& $python -m pytest tests/unit/test_compare_proxy_stages.py tests/unit/test_runtime_fingerprint.py tests/unit/test_verify_frozen_inputs.py tests/unit/test_run_proxy.py tests/unit/test_benchmark_release.py -q
& $ruff check tools/compare_proxy_stages.py tools/runtime_fingerprint.py tools/verify_frozen_inputs.py tools/run_proxy.py tools/benchmark_release.py tests/unit/test_compare_proxy_stages.py tests/unit/test_runtime_fingerprint.py tests/unit/test_verify_frozen_inputs.py tests/unit/test_run_proxy.py tests/unit/test_benchmark_release.py
& $verifyFrozen
git diff --check
git add docs/superpowers/plans/2026-08-27-compasscart-balanced-score-hardening.md tools/compare_proxy_stages.py tools/runtime_fingerprint.py tools/verify_frozen_inputs.py tools/run_proxy.py tools/benchmark_release.py tests/unit/test_compare_proxy_stages.py tests/unit/test_runtime_fingerprint.py tests/unit/test_verify_frozen_inputs.py tests/unit/test_run_proxy.py tests/unit/test_benchmark_release.py reports/final/balanced-score-hardening-results.md
git commit -m "test: enforce balanced proxy stage gates"
```

### Task 2: S1 Clarification Alias Gating

**Files:**

- Modify: `src/compasscart/parser.py`
- Modify: `tests/unit/test_parser.py`
- Test: `tests/unit/test_state.py`
- Test: `tests/integration/test_functional_edge_cases.py`

- [ ] **Step 1: Add failing clarification-boundary tests**

Add tests proving:

- pending `style` accepts `adjustable` as style when the vocabulary supports it;
- pending `size` plus terse `boots` creates neither category hard constraint nor `size=boots` soft constraint;
- a fixed cross-attribute value such as `Blue.` is suppressed during an unrelated pending clarification;
- explicit `category: boots`, `size: adjustable`, and `style: adjustable` cues survive;
- `Actually, I need boots` remains a real goal/category override;
- explicit negative preferences remain recognized;
- `no preference` rejects only the pending attribute and preserves route/goal.

Run the named new tests and confirm they fail for the intended fixed-alias bypass and residual-text reasons.

- [ ] **Step 2: Implement one gate for fixed and dynamic aliases**

Change `_alias_allowed()` so `expected_attribute` is the default boundary for every fixed and dynamic alias. Permit a cross-attribute match only for an explicit cue for that attribute, an explicit goal/category replacement, a standard override expression, or an explicit negative preference.

Extract one vocabulary-span helper and use it both for matching and `_has_unrecognized_expected_text()`. A known alias suppressed by the pending gate still counts as a recognized span, so it cannot be reinterpreted as a soft value for the pending attribute.

Do not add catalog IDs, sample rules, route changes, or retrieval behavior.

- [ ] **Step 3: Run focused and full semantic tests**

```powershell
& $python -m pytest tests/unit/test_parser.py tests/unit/test_state.py tests/integration/test_functional_edge_cases.py tests/integration/test_scenarios.py -q
& $ruff check src/compasscart/parser.py tests/unit/test_parser.py
```

- [ ] **Step 4: Request specification review, then quality review**

The specification reviewer checks only design sections 5.2, 8, 9.3-9.4, and the stage gates. After specification approval, the quality reviewer checks parser over-gating, Unicode/word boundaries, deterministic parsing, and test coverage. Resolve every finding, rerun Step 3, and freeze the resulting runtime fingerprint before any S1 score run.

- [ ] **Step 5: Run the S1 development suite, then one stress veto**

```powershell
& $verifyFrozen
& $python -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite representative --folds 1 2 3 4 --output var/balanced-hardening/proxy-v1/dev-s1.json
& $python -m tools.compare_proxy_stages --phase development --parent-dev var/balanced-hardening/proxy-v1/dev-p0.json --candidate-dev var/balanced-hardening/proxy-v1/dev-s1.json --stage S1 --output var/balanced-hardening/proxy-v1/gate-s1-development.json
# Only after the preceding command accepts:
& $verifyFrozen
& $python -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite stress --output var/balanced-hardening/proxy-v1/stress-s1.json
& $verifyFrozen
& $python -m tools.benchmark_release --catalog data/catalog.jsonl --transcript var/balanced-hardening/benchmark-transcript.jsonl --trials 3 --cwd-mode outside --output var/balanced-hardening/benchmark-s1.json --compare var/balanced-hardening/benchmark-r0-wall-v2.json --comparison-mode resource
& $python -m tools.compare_proxy_stages --phase complete --development-receipt var/balanced-hardening/proxy-v1/gate-s1-development.json --parent-dev var/balanced-hardening/proxy-v1/dev-p0.json --candidate-dev var/balanced-hardening/proxy-v1/dev-s1.json --parent-stress var/balanced-hardening/proxy-v1/stress-p0.json --candidate-stress var/balanced-hardening/proxy-v1/stress-s1.json --resource-report var/balanced-hardening/benchmark-s1.json --stage S1 --output var/balanced-hardening/proxy-v1/gate-s1.json
```

Expected: accepted with zero fallback/invalid, or reject and restore only S1 changes. Do not run audit.

- [ ] **Step 6: Recheck the scored fingerprint**

After a development receipt exists, do not change production source or default config. Immediately before commit, recompute both fingerprints and require they match the complete receipt. Any mismatch, including a review or cleanup claimed to be behavior-neutral, invalidates all S1 reports and requires new filenames plus development, stress, resource, and complete comparison again.

- [ ] **Step 7: Commit only an accepted S1**

```powershell
git add src/compasscart/parser.py tests/unit/test_parser.py tests/unit/test_state.py tests/integration/test_functional_edge_cases.py reports/final/balanced-score-hardening-results.md
git commit -m "fix: gate clarification aliases by pending attribute"
```

### Task 3: S2 Substantive Query History

**Files:**

- Modify: `src/compasscart/parser.py`
- Modify: `src/compasscart/state.py`
- Modify: `src/compasscart/agent.py`
- Modify: `tests/unit/test_parser.py`
- Modify: `tests/unit/test_state.py`
- Modify: `tests/contract/test_agent_contract.py`
- Test: `tests/integration/test_scenarios.py`

- [ ] **Step 1: Add failing evidence-classification and history tests**

Add an internal, backward-compatible `ParseResult.has_substantive_evidence` signal. Tests require `No preference.`, continuation/request-more messages, and exact control-only templates to be false; unknown product text such as `with a magnetic clasp` is true.

Test that filtered messages never enter `query_history`, active constraints still enter query text, `_query_text()` does not reinsert the raw control message, the last four substantive messages remain bounded, and a true goal override clears old history while profile constraints remain.

- [ ] **Step 2: Implement strict control-only classification**

Default unknown non-empty product text to substantive. Mark only recognized no-preference, continuation, and full-message control intents non-substantive. Do not use loose keyword substring filters such as `more`, `thanks`, or `search` inside a longer shopping request.

In `SessionStore.update()`, append the raw message only when the parse result is substantive. Preserve the existing goal-version reset order. In `Agent._query_text()`, compose only bounded state history plus active constraints; keep the current signature for compatibility but do not add the raw message separately.

- [ ] **Step 3: Verify semantics and contract**

```powershell
& $python -m pytest tests/unit/test_parser.py tests/unit/test_state.py tests/contract/test_agent_contract.py tests/integration/test_scenarios.py tests/integration/test_functional_edge_cases.py tests/integration/test_fallbacks.py -q
& $ruff check src/compasscart/parser.py src/compasscart/state.py src/compasscart/agent.py tests/unit/test_parser.py tests/unit/test_state.py tests/contract/test_agent_contract.py
```

- [ ] **Step 4: Request specification review, then quality review**

The specification reviewer checks design sections 5.3 and 9.5-9.6. The quality reviewer checks false-positive control filtering, reset order, raw-message reinsertion, and contract compatibility. Resolve every finding, rerun Step 3, and freeze the resulting runtime/config fingerprints before scoring.

- [ ] **Step 5: Run S2 development and one stress veto against accepted S1**

Use the accepted S1 reports as parents. If S1 was rejected, use the unchanged P0 reports and name that fact in the results report.

```powershell
& $verifyFrozen
& $python -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite representative --folds 1 2 3 4 --output var/balanced-hardening/proxy-v1/dev-s2.json
& $python -m tools.compare_proxy_stages --phase development --parent-dev <accepted-parent-dev> --candidate-dev var/balanced-hardening/proxy-v1/dev-s2.json --stage S2 --output var/balanced-hardening/proxy-v1/gate-s2-development.json
# Only after development acceptance, run stress and the complete comparator:
& $verifyFrozen
& $python -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite stress --output var/balanced-hardening/proxy-v1/stress-s2.json
& $verifyFrozen
& $python -m tools.benchmark_release --catalog data/catalog.jsonl --transcript var/balanced-hardening/benchmark-transcript.jsonl --trials 3 --cwd-mode outside --output var/balanced-hardening/benchmark-s2.json --compare var/balanced-hardening/benchmark-r0-wall-v2.json --comparison-mode resource
& $python -m tools.compare_proxy_stages --phase complete --development-receipt var/balanced-hardening/proxy-v1/gate-s2-development.json --parent-dev <accepted-parent-dev> --candidate-dev var/balanced-hardening/proxy-v1/dev-s2.json --parent-stress <accepted-parent-stress> --candidate-stress var/balanced-hardening/proxy-v1/stress-s2.json --resource-report var/balanced-hardening/benchmark-s2.json --stage S2 --output var/balanced-hardening/proxy-v1/gate-s2.json
```

- [ ] **Step 6: Recheck fingerprints and commit an accepted S2**

After the development receipt, no production/config change is allowed. Recompute current fingerprints and require the complete receipt match; any mismatch invalidates and reruns all S2 evidence. Commit only after the reviews and all gates pass:

```powershell
git add src/compasscart/parser.py src/compasscart/state.py src/compasscart/agent.py tests/unit/test_parser.py tests/unit/test_state.py tests/contract/test_agent_contract.py reports/final/balanced-score-hardening-results.md
git commit -m "fix: retain only substantive shopping evidence"
```

### Task 4: S3 Bounded Ranking Calibration

**Files:**

- Modify: `src/compasscart/models.py`
- Modify: `src/compasscart/config.py`
- Modify: `src/compasscart/retrieval.py`
- Modify: `src/compasscart/ranker.py`
- Modify: `src/compasscart/agent.py`
- Modify: `tests/unit/test_models.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_retrieval.py`
- Modify: `tests/unit/test_ranker.py`
- Test: `tests/integration/test_scenarios.py`

- [ ] **Step 1: Add failing compatible evidence tests**

Append defaulted internal fields to `Candidate`:

```python
source_ranks: dict[str, int] = field(default_factory=dict)
pre_rank: int | None = None
```

Tests require Retriever to preserve each positive-weight non-duplicate source rank and deterministic fused pre-rank. Fallback candidates may keep empty ranks and `None`. Ranker-created candidates must copy both fields. Existing positional construction and official response serialization must remain compatible.

- [ ] **Step 2: Add failing bounded-feature tests**

Add config validation and ranker tests for these predeclared values only:

| Factor | Parent/default | Development candidates |
| --- | ---: | --- |
| attribute weight | `0.00` | `0.05`, `0.10` |
| consensus bonus | `0.00` | `0.025`, `0.05` |
| fusion weight | `0.10` | `0.15` |
| boundary bonus | `0.00` | `0.025` |
| MMR lambda | `0.85` | fixed; never test `1.0` |
| adaptive Browsing MMR | `False` | `True` with the fixed gates below |

The lexical+dense+fusion+attribute source budget is always exactly `0.40`:

```python
source_weight = (0.40 - fusion_weight - attribute_weight) / 2.0
```

Reject negative weights, values outside the table, or `fusion + attribute > 0.40`. Consensus requires at least two positive non-profile sources and at least one of lexical/attribute. Profile-only evidence never counts. Consensus and boundary have separate bounded caps: at most `0.05` consensus plus `0.025` boundary, so total auxiliary evidence can never exceed `0.075`. This keeps the boundary experiment measurable even if development selects consensus `0.05`.

Boundary bonus requires all of: `pre_rank <= 10`, exact candidate, no hard conflict, and consensus. It is a score feature, never a reserved slot. Tests prove a constructed 11th-ranked candidate can enter Top 10 when warranted, but relaxed candidates remain after every exact candidate, hard conflicts cannot leapfrog satisfying candidates, and ties end by ascending ID.

- [ ] **Step 3: Implement evidence propagation with neutral defaults**

Compute source ranks from already-bounded source lists and `pre_rank` from the fused order. Preserve fields through ranking. Add config fields with behavior-neutral defaults and wire them through `Agent`.

Normalize the attribute source exactly like lexical/dense. Retain the Retriever's route-weighted fused score rather than rebuilding route weights in Ranker. Apply independent consensus and boundary caps after calculating eligibility; add an explicit test that an eligible candidate may receive `0.05 + 0.025 = 0.075`, never more. Keep the existing `-0.60` hard-conflict penalty and `(relaxed, -score, parent_asin)` ordering.

- [ ] **Step 4: Make MMR adaptive behind fixed gates**

Do not search MMR thresholds. Add `adaptive_browsing_mmr: bool = False`; only the `True` candidate activates this behavior. For Browsing only, apply MMR to exact and relaxed segments separately when a segment has at least 11 candidates, the normalized score gap between positions 10 and 11 is at most `0.025`, and at least three pairs among the first ten candidates have attribute-term Jaccard similarity at least `0.60`. Otherwise return the base deterministic order. Preserve the deadline skip and diagnostic.

Tests cover gate-off equivalence, gate-on determinism, deadline skip, no exact/relaxed crossing, and no `lambda=1.0` path.

- [ ] **Step 5: Verify the neutral infrastructure**

```powershell
& $python -m pytest tests/unit/test_models.py tests/unit/test_config.py tests/unit/test_retrieval.py tests/unit/test_ranker.py tests/integration/test_scenarios.py -q
& $ruff check src/compasscart/models.py src/compasscart/config.py src/compasscart/retrieval.py src/compasscart/ranker.py src/compasscart/agent.py tests/unit/test_models.py tests/unit/test_config.py tests/unit/test_retrieval.py tests/unit/test_ranker.py
```

- [ ] **Step 6: Complete specification and quality review**

The specification reviewer checks design sections 5.4, 6, 7, 9.7-9.9, and all stage gates. The quality reviewer checks budget arithmetic, source-rank provenance, fallback candidates, deterministic ties, exact/relaxed ordering, hard-conflict safety, MMR complexity, and field-copy completeness. Resolve every finding and rerun Step 5 before any factor scoring. Freeze the neutral runtime fingerprint; each parameter-only candidate must have its own config fingerprint.

- [ ] **Step 7: Evaluate S3 factors one at a time**

Start from the accepted semantic parent. Before every run, call `& $verifyFrozen`. For each factor in the table order, keep prior accepted factors fixed, run all listed values on development only, and create a development receipt with `--phase development --stage S3-<factor>`. Choose by development score and the deterministic smaller-change tie-break. Then run stress once for the chosen value, run one three-trial `--comparison-mode resource` benchmark against `benchmark-r0-wall-v2.json`, and call complete comparison with the selected development receipt and resource report. Candidate dev/stress/resource runtime and config fingerprints must match. If stress or resource fails, retain that factor's parent and do not test its runner-up.

Use filenames such as:

```text
dev-s3-attribute-005.json
dev-s3-attribute-010.json
stress-s3-attribute-005.json
gate-s3-attribute-005.json
benchmark-s3-attribute-005.json
```

Repeat for consensus, fusion, boundary, and adaptive MMR (`False` versus `True`). A factor that cannot achieve `+0.003` selection while passing every development gate is rejected before stress. Every accepted factor must pass its resource gate before the next factor starts. Do not inspect or run audit.

- [ ] **Step 8: Recheck fingerprints and commit**

After a factor's development receipt, do not change production source. Require the complete receipt to match the current runtime/config fingerprints. Any mismatch invalidates that factor's development, stress, and resource evidence.

Commit the neutral evidence plumbing plus only accepted default values/features:

```powershell
git add src/compasscart/models.py src/compasscart/config.py src/compasscart/retrieval.py src/compasscart/ranker.py src/compasscart/agent.py tests/unit/test_models.py tests/unit/test_config.py tests/unit/test_retrieval.py tests/unit/test_ranker.py reports/final/balanced-score-hardening-results.md
git commit -m "feat: calibrate bounded rank evidence"
```

### Task 5: S4 Answerability-Aware Questions

**Files:**

- Modify: `src/compasscart/parser.py`
- Modify: `src/compasscart/question_policy.py`
- Modify: `src/compasscart/agent.py`
- Modify: `tests/unit/test_parser.py`
- Modify: `tests/unit/test_question_policy.py`
- Modify: `tests/contract/test_agent_contract.py`
- Test: `tests/integration/test_scenarios.py`

- [ ] **Step 1: Add failing support and policy tests**

Add a side-effect-free Parser support method for `(attribute, value)` and inject two callbacks into `QuestionPolicy`: parser support and catalog retrieval support. Tests require a high-reduction unsupported attribute to lose to a lower-reduction supported attribute; catalog-only or parser-only support is insufficient.

Require at least two supported meaningful partitions. A partition is meaningful when it has at least two candidate IDs or at least `0.05` normalized probability mass. Keep asked, rejected/no-preference, and explicit hard-constrained attributes blocked. Continuation, at most ten candidates, or turn 10 must return no question.

- [ ] **Step 2: Implement fixed utility semantics**

Use supported partition mass, without learning from public outcomes:

```text
utility = candidate_reduction
          * answerability
          * parser_support
          * retrieval_support
          * remaining_turn_value
          - no_preference_risk
```

Use the existing fixed response-likelihood table for answerability. `parser_support` and `retrieval_support` are the covered probability-mass fractions before their respective filters. Set `no_preference_risk = 0.05 * (1 - answerability)`. Do not add route-specific numeric bonuses.

Use fixed late thresholds: turns 1-7 `0.0`, turn 8 `0.10`, turn 9 `0.15`, turn 10 no question. Buying and `override_scope != "none"` affect only the deterministic tie-break, preferring attributes capable of representing a hard requirement after equal utility; the fixed attribute order is the final tie-break.

Recommendations are always returned even when a question is asked.

- [ ] **Step 3: Verify policy behavior**

```powershell
& $python -m pytest tests/unit/test_parser.py tests/unit/test_question_policy.py tests/contract/test_agent_contract.py tests/integration/test_scenarios.py -q
& $ruff check src/compasscart/parser.py src/compasscart/question_policy.py src/compasscart/agent.py tests/unit/test_parser.py tests/unit/test_question_policy.py
```

- [ ] **Step 4: Complete specification and quality review**

The specification reviewer checks design sections 5.5, 9.10, and stage gates. The quality reviewer checks callback side effects, probability normalization, unsupported partitions, stable tie-breaking, turn thresholds, and recommendation preservation. Resolve findings and rerun Step 3 before freezing the S4 runtime/config fingerprints.

- [ ] **Step 5: Run one fixed S4 candidate and apply scoring gates**

Call `& $verifyFrozen`, then run representative development against the accepted S3 parent and create `gate-s4-development.json`. Only if it passes development, call the verifier again and run stress once. Call the verifier again, run `benchmark_release --comparison-mode resource` as `benchmark-s4.json`, and require complete comparison to bind the development receipt plus matching dev/stress/resource runtime and config fingerprints. Use `dev-s4.json`, `stress-s4.json`, and `gate-s4.json`; S4 uses the scoring threshold and the fixed utility constants are never tuned on stress or resource evidence.

- [ ] **Step 6: Recheck fingerprints and commit an accepted S4**

After the development receipt, no production/config change is allowed. Recompute current fingerprints and require the complete receipt to match; otherwise rerun all S4 evidence.

```powershell
git add src/compasscart/parser.py src/compasscart/question_policy.py src/compasscart/agent.py tests/unit/test_parser.py tests/unit/test_question_policy.py tests/contract/test_agent_contract.py reports/final/balanced-score-hardening-results.md
git commit -m "feat: select answerable retrieval questions"
```

If rejected, retain the S3 parent and record S4 as rejected; S5 still starts from the last accepted parent.

### Task 6: S5 Adaptive Attribute Depth

**Files:**

- Modify: `src/compasscart/config.py`
- Modify: `src/compasscart/retrieval.py`
- Modify: `src/compasscart/agent.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_retrieval.py`
- Test: `tests/performance/test_runtime.py`

- [ ] **Step 1: Add failing trigger and bound tests**

Keep the base attribute depth at 150. Add a maximum depth accepting only 150, 200, or 250. Test:

- Browsing without hard constraints uses 150 only;
- Buying or explicit hard constraints may expand when exact results are short or the 150 boundary is saturated;
- exact sufficiency without saturation does not expand;
- an expired deadline prevents expansion and records `attribute_depth_budget`;
- effective maximum depth is capped by `plan.candidate_limit`;
- lexical, Dense, and profile limits remain 150 in all cases;
- exact/relaxed/fallback ordering and valid unique IDs remain unchanged.

- [ ] **Step 2: Implement two-pass attribute-only expansion**

Request exactly the existing base limit of 150 attribute matches. Treat `len(base_attribute_ids) == 150` as the conservative truncation-boundary signal; never request item 151 merely to probe. When `attribute_max_depth == 150`, no attribute call may exceed 150. After initial exact fusion, expand and recompute fusion only when all conditions hold:

1. route is Buying or explicit hard constraints exist;
2. fewer than the desired exact results exist or the base probe is saturated;
3. deadline is absent or still has time;
4. `effective_attribute_limit = min(config.attribute_max_depth, plan.candidate_limit)` exceeds 150.

Use `effective_attribute_limit` for the expanded attribute call, including the tested `max=250, candidate_limit=200 -> 200` case. Do not re-run or enlarge lexical, Dense, or profile retrieval. Keep diagnostics internal and response schema unchanged.

- [ ] **Step 3: Verify focused tests**

```powershell
& $python -m pytest tests/unit/test_config.py tests/unit/test_retrieval.py tests/performance/test_runtime.py tests/integration/test_fallbacks.py -q
& $ruff check src/compasscart/config.py src/compasscart/retrieval.py src/compasscart/agent.py tests/unit/test_config.py tests/unit/test_retrieval.py
```

- [ ] **Step 4: Complete specification and quality review**

The specification reviewer checks design sections 5.6, 8, 9.11, and resource gates. The quality reviewer checks double-search work, deadline behavior, source-limit isolation, deterministic recomputation, base-call limit 150, and candidate-limit edges. Resolve findings and rerun Step 3 before scoring.

- [ ] **Step 5: Select maximum depth on development only**

Before every run call `& $verifyFrozen`. Run development candidates 200 and 250 against the last accepted parent and produce a development receipt for each. Select the higher development score, tie-breaking to 200, only if the full development gate passes. Then run stress once for that selected depth and a three-trial `--comparison-mode resource` benchmark against R0. Complete comparison must bind the selected receipt and matching dev/stress/resource runtime and config fingerprints. Stress or resource failure rejects S5 entirely; neither can select the other depth.

Use `dev-s5-depth-200.json`, `dev-s5-depth-250.json`, one matching stress file, and a gate report. Do not run audit.

- [ ] **Step 6: Recheck fingerprints and commit an accepted S5**

After the development receipt, no production/config change is allowed. Recompute current fingerprints and require the complete receipt to match; otherwise rerun all S5 evidence.

```powershell
git add src/compasscart/config.py src/compasscart/retrieval.py src/compasscart/agent.py tests/unit/test_config.py tests/unit/test_retrieval.py tests/performance/test_runtime.py reports/final/balanced-score-hardening-results.md
git commit -m "feat: adapt bounded attribute recall depth"
```

### Task 7: Freeze the Candidate and Run Pre-Audit Verification

**Files:**

- Modify: `tools/run_agent.py`
- Create: `tests/unit/test_run_agent.py`
- Modify: `reports/final/balanced-score-hardening-results.md`
- Modify only if metrics are stale: `README.md`, `reports/final/final-results.json`

- [ ] **Step 1: Freeze configuration and record accepted lineage**

Record each stage parent/candidate commit, configuration hash, dev/stress aggregate deltas, gate report SHA-256, and rejection reason counts. Do not include proxy target/session IDs. From this point, no scoring behavior or parameter may change.

Before freezing, add a test-backed evidence mode to `tools/run_agent.py`: construct one `Agent`, call the unchanged official `evaluate()` exactly once, then summarize trace count, any non-empty top-level trace `fallbacks`, and trace/call-count consistency into a separate aggregate evidence object. The official score JSON remains byte-for-schema compatible with evaluator output. The wrapper must fail if evaluation returns without one trace per attempted response or any trace contains a fallback. This is development tooling only and cannot affect `agent.Agent` behavior.

- [ ] **Step 2: Run all static and functional verification**

```powershell
& $python -m pytest -q
& $ruff check src tests tools agent.py
git diff --check
git status --short
```

Expected: all tests pass, Ruff/diff clean, and only the three protected files remain untracked outside intentional final-report edits.

- [ ] **Step 3: Verify all nine frozen hashes**

Run `& $verifyFrozen`, then independently spot-check with `Get-FileHash -Algorithm SHA256 -LiteralPath ...` using the exact guardrail paths. Any mismatch stops the release before audit.

- [ ] **Step 4: Run the final full-catalog resource benchmark**

```powershell
& $verifyFrozen
& $python -m tools.benchmark_release --catalog data/catalog.jsonl --transcript var/balanced-hardening/benchmark-transcript.jsonl --trials 3 --cwd-mode outside --output var/balanced-hardening/benchmark-final.json --compare var/balanced-hardening/benchmark-r0-wall-v2.json --comparison-mode resource
```

Require Dense available in all trials, fallback zero, P95/init/peak medians no more than 5% worse than R0, and every response below 1.5 seconds. Ranking output may differ from R0; output validity and deterministic repeated-trial hashes must hold.

- [ ] **Step 5: Run final specification and code review**

Review the complete accepted diff from foundation commit `ac36def` through the frozen candidate. Resolve only correctness, contract, security, determinism, or documented-gate issues. Any behavior change invalidates the candidate freeze and requires new development/stress reports before returning here. Audit and public evaluation remain untouched.

- [ ] **Step 6: Commit the immutable pre-audit candidate**

```powershell
git add src/compasscart/parser.py src/compasscart/state.py src/compasscart/agent.py src/compasscart/models.py src/compasscart/config.py src/compasscart/retrieval.py src/compasscart/ranker.py src/compasscart/question_policy.py tests/unit/test_parser.py tests/unit/test_state.py tests/unit/test_models.py tests/unit/test_config.py tests/unit/test_retrieval.py tests/unit/test_ranker.py tests/unit/test_question_policy.py tests/contract/test_agent_contract.py tests/integration/test_scenarios.py tests/integration/test_functional_edge_cases.py tests/integration/test_fallbacks.py tests/performance/test_runtime.py tools/run_agent.py tests/unit/test_run_agent.py reports/final/balanced-score-hardening-results.md README.md reports/final/final-results.json
git commit -m "release: freeze balanced score candidate"
```

### Task 8: Final Audit Once and Public Score Once

**Files:**

- Output only: `var/balanced-hardening/public-final-<short-commit>.json` (raw official result; ignored, never packaged)
- Create: `reports/final/score-results-<short-commit>-2026-08-27.json` (sanitized aggregate only)
- Create: `reports/final/score-results-<short-commit>-2026-08-27.md`
- Modify: `reports/final/balanced-score-hardening-results.md`
- Modify: `README.md`
- Modify: `reports/final/final-results.json`

- [ ] **Step 1: Preflight the one-use audit path without evaluating**

Confirm the candidate commit is clean except protected untracked files, all Task 7 evidence is accepted, `var/balanced-hardening/proxy-v1/audit/final.json` and `.lock` do not exist, and the baseline audit aggregate remains present. Do not read any audit session data; the sealed audit writer emits aggregate-only output.

- [ ] **Step 2: Run the final proxy audit exactly once**

```powershell
& $verifyFrozen
& $python -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite representative --audit-label final --output var/balanced-hardening/proxy-v1/audit/final.json
```

Compare aggregate-only baseline/final values. Require final TechnicalScore at least `0.500563`; Buying/Browsing/Intent Override HitRate declines no more than `0.025`; Boundary hits do not decline; each scenario TechnicalScore declines no more than `0.03`; fallback and invalid responses remain zero. If rejected, stop the release, do not tune from audit, and do not run another audit.

- [ ] **Step 3: Run the official public evaluation exactly once**

Only after the audit passes:

```powershell
& $verifyFrozen
& $python -m tools.run_agent --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output var/balanced-hardening/public-final-<short-commit>.json --evidence-output var/balanced-hardening/public-final-evidence.json
```

Require valid output, trace fallback count zero, complete call/trace evidence, and TechnicalScore at least `0.655411`. Report TechnicalScore, HitRate@10, MRR, MTTC, efficiency, all four scenario aggregates, and recover/regress counts against the prior public result. Generate `reports/final/score-results-<short-commit>-2026-08-27.json` from an explicit aggregate allowlist; it must contain no `sessions`, `sample_id`, recommendations, target IDs, or intent data. Keep `reports/final/final-results.json` aggregate-only under the same restriction because the competition package includes it. Do not use the outcome to tune or run the public evaluator again.

If the public threshold fails, record the sanitized aggregate result and stop release publication. Do not enter Task 9, build new release ZIPs, or push GitHub `main`.

- [ ] **Step 4: Write the Chinese final score summary**

The Markdown report distinguishes:

- proxy development selection evidence;
- stress veto evidence;
- one aggregate-only audit result;
- one official public result;
- runtime benchmark;
- frozen hashes and test results;
- remaining opportunities that were rejected or intentionally deferred;
- `macOS verification pending` unless a real Apple Silicon runner was used.

Do not claim an estimated score as official and do not claim cross-platform completion from Windows evidence.

### Task 9: Competition Package, Source Bundle, and GitHub Main

**Files:**

- Create: `tools/package_source.py`
- Create: `tests/unit/test_package_source.py`
- Modify: `reports/final/balanced-score-hardening-results.md`
- Output only: `dist/compasscart-submission.zip`
- Output only: `dist/compasscart-source.zip`
- Output only: `dist/SHA256SUMS`

- [ ] **Step 1: Write a failing explicit-allowlist source-package test**

The source bundle contains root entry/config/attribution files, `src/compasscart`, `tests`, development `tools`, documentation, sanitized aggregate Markdown reports, sanitized aggregate score JSON, licenses, and required runtime assets. It excludes `.git`, `.venv`, caches, `var`, `dist`, the evaluator, public labels, organizer catalog, every raw/session-level score result, secrets, and the three protected files. ZIP timestamps and ordering are fixed for reproducible bytes.

```python
def test_source_bundle_contains_code_and_excludes_competition_labels(): ...
def test_source_bundle_is_reproducible_and_secret_scan_clean(): ...
def test_packaged_final_reports_reject_sessions_and_sample_ids(): ...
```

- [ ] **Step 2: Implement and verify source packaging**

Reuse `tools.package_submission` constants, secret scan, asset verification, and fixed timestamp where possible. Keep a separate explicit source allowlist; never glob the repository root.

```powershell
& $python -m pytest tests/unit/test_package_source.py tests/contract/test_submission_package.py -q
& $ruff check tools/package_source.py tests/unit/test_package_source.py
```

- [ ] **Step 3: Commit final reports and packaging code**

```powershell
git add tools/package_source.py tests/unit/test_package_source.py README.md reports/final/final-results.json reports/final/balanced-score-hardening-results.md reports/final/score-results-<short-commit>-2026-08-27.json reports/final/score-results-<short-commit>-2026-08-27.md
git commit -m "release: publish balanced hardening evidence"
```

Verify `git status --short` shows only the three protected untracked files.

- [ ] **Step 4: Build both deliverables from the final commit**

```powershell
& $verifyFrozen
& $python -m tools.package_submission
& $python -m tools.package_source
& $python -m pytest tests/contract/test_submission_package.py tests/unit/test_package_source.py -q
Get-FileHash -Algorithm SHA256 -LiteralPath dist/compasscart-submission.zip,dist/compasscart-source.zip
```

Write both hashes to `dist/SHA256SUMS`. Label `compasscart-submission.zip` as the competition upload and `compasscart-source.zip` as the code-integrated project bundle. The source bundle is not a valid competition upload.

- [ ] **Step 5: Update GitHub `main` without force**

The user has authorized the repository update. First fetch and verify ancestry:

```powershell
git fetch origin
git log --oneline --decorate -5
git merge-base --is-ancestor origin/main HEAD
git status --short
```

If `origin/main` advanced, integrate it non-destructively and rerun package/tests; never force-push. When ancestry and verification pass:

```powershell
git push origin HEAD:main
git ls-remote --heads origin main
```

Confirm the remote `main` SHA equals local `HEAD`. Do not stage or upload the protected untracked files or `var/` artifacts.

- [ ] **Step 6: Final delivery check**

Report the final commit, GitHub `main` SHA, official score delta, audit delta, test count, benchmark deltas, both package paths and SHA-256 values, and remaining macOS status. Do not claim completion until the push, package contracts, and hashes are verified.

---

## Stop Conditions

Stop before audit if any development/stress, functional, integrity, or resource gate fails and cannot be corrected without changing the frozen candidate. Stop after a failed final audit without another audit attempt. Stop after the one public run without parameter tuning or a second run. A public score below `0.655411` also stops package publication and the GitHub `main` push. Never relax a gate merely to reach the aspirational `0.68-0.70` range.
