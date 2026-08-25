# CompassCart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, offline-first shopping agent that beats the official BM25 baseline across Buying, Browsing, Intent Override, and Boundary sessions while always satisfying the official Agent contract.

**Architecture:** Keep the standard-library lexical path as the always-available core, then add versioned dialog state, route-aware retrieval, conversion-aware clarification, and an optional local ONNX dense retriever behind narrow interfaces. Every advanced component has a tested fallback, and official scoring never requires network access or API credentials.

**Tech Stack:** Python 3.10+, standard library (`sqlite3`, `dataclasses`, `json`, `re`), optional NumPy/ONNX Runtime/tokenizers for dense retrieval, pytest for tests, official deterministic evaluator.

---

## File Map

```text
agent.py                              Official submission entry point only
requirements.txt                     Minimal optional runtime dependencies
requirements-dev.txt                 Test and lint dependencies
requirements-assets.txt              Heavy one-time dense asset dependencies
src/compasscart/__init__.py           Public package exports
src/compasscart/config.py             Immutable runtime configuration
src/compasscart/models.py             Shared dataclasses and protocols
src/compasscart/normalization.py      Text and catalog field normalization
src/compasscart/catalog.py            Catalog load, FTS5, attributes, fallbacks
src/compasscart/parser.py             Intent, slot, and override parsing
src/compasscart/state.py              SessionStore and Constraint Ledger
src/compasscart/router.py             Buying/Browsing route decisions
src/compasscart/retrieval.py          Candidate generation and RRF fusion
src/compasscart/ranker.py             Constraint-aware ranking and diversity
src/compasscart/question_policy.py    Expected conversion-gain questions
src/compasscart/response.py           Contract-safe response construction
src/compasscart/dense.py              Optional local ONNX dense backend
src/compasscart/tracing.py            Bounded structured trace records
src/compasscart/agent.py              Runtime orchestration
tools/download_kit.ps1                Reproducible official-kit import
tools/__init__.py                      Makes development tools module-runnable
tools/run_agent.py                     Official-evaluator-compatible custom Agent runner
tools/build_dense_assets.py           Offline model/vector asset generation
tools/run_cv.py                        Stratified development-fold evaluation
tools/analyze_failures.py              Scenario and failure summaries
tools/package_submission.py           Clean submission bundle creation
tests/fixtures/catalog.jsonl          Small deterministic test catalog
tests/unit/                            Focused component tests
tests/integration/                     Multi-turn scenario tests
tests/contract/                        Official API contract tests
tests/performance/                     Latency and leak regression tests
```

The official `evaluator/`, `data/public_set.jsonl`, starter, competition docs, and baseline files are imported unchanged from `TechJam2026/techjam-conversational-search`.

## Task 1: Import the Official Kit and Reproduce Baseline

**Files:**
- Create: `tools/download_kit.ps1`
- Import unchanged: `.gitignore`, `DATA_ATTRIBUTION.md`, `data/`, `docs/agent_api_contract.json`, `docs/baseline_results.json`, `docs/competition_specification.md`, `docs/evaluation_config.json`, `docs/submission_rules.md`, `evaluator/`, `starter/`, `tests/test_evaluator.py`
- Create locally, ignored by Git: `data/catalog.jsonl`

- [ ] **Step 1: Add the official repository as a read-only remote**

Run:

```powershell
git remote add official https://github.com/TechJam2026/techjam-conversational-search.git
git fetch official main
```

Expected: `official/main` resolves successfully without changing the working tree.

- [ ] **Step 2: Import only participant-visible files**

Run each command separately:

```powershell
git checkout official/main -- .gitignore
git checkout official/main -- DATA_ATTRIBUTION.md
git checkout official/main -- data
git checkout official/main -- evaluator
git checkout official/main -- starter
git checkout official/main -- tests
git checkout official/main -- docs/agent_api_contract.json docs/baseline_results.json docs/competition_specification.md docs/evaluation_config.json docs/submission_rules.md
```

Expected: no `organizer/` directory and the existing `docs/superpowers/` files remain intact.

- [ ] **Step 3: Create a reproducible catalog downloader**

Create `tools/download_kit.ps1` with:

```powershell
$ErrorActionPreference = "Stop"
$releaseBase = "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit"
$dataDir = Join-Path $PSScriptRoot "..\data"
$archive = Join-Path $dataDir "catalog.jsonl.gz"
$checksumFile = Join-Path $dataDir "SHA256SUMS"
$catalog = Join-Path $dataDir "catalog.jsonl"

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
Invoke-WebRequest "$releaseBase/catalog.jsonl.gz" -OutFile $archive
Invoke-WebRequest "$releaseBase/SHA256SUMS" -OutFile $checksumFile

$expected = ((Get-Content $checksumFile | Where-Object { $_ -match "catalog.jsonl.gz" }) -split "\s+")[0].ToLowerInvariant()
$actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
if ($expected -ne $actual) { throw "catalog checksum mismatch" }

$inputStream = [System.IO.File]::OpenRead($archive)
$outputStream = [System.IO.File]::Create($catalog)
$gzipStream = [System.IO.Compression.GZipStream]::new(
    $inputStream,
    [System.IO.Compression.CompressionMode]::Decompress
)
try {
    $gzipStream.CopyTo($outputStream)
} finally {
    $gzipStream.Dispose()
    $outputStream.Dispose()
    $inputStream.Dispose()
}
Write-Host "Catalog ready at $catalog"
```

- [ ] **Step 4: Download, verify, and decompress the catalog**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File tools/download_kit.ps1
```

Expected: `data/catalog.jsonl` exists, SHA256 validation succeeds, and the script prints `Catalog ready`.

- [ ] **Step 5: Run the official baseline**

Run:

```powershell
& "C:\Users\renha\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m evaluator.local_evaluator
```

Expected aggregate metrics: HitRate@10 `0.125`, MRR `0.068034`, MTTC `9.81`, TechnicalScore `0.10671`.

- [ ] **Step 6: Commit the participant kit import**

```powershell
git add .gitignore DATA_ATTRIBUTION.md data docs evaluator starter tests tools/download_kit.ps1
git commit -m "chore: import official Track 4 participant kit"
```

## Task 2: Bootstrap Package, Config, and Shared Models

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `requirements-assets.txt`
- Create: `src/compasscart/__init__.py`
- Create: `src/compasscart/config.py`
- Create: `src/compasscart/models.py`
- Create: `tests/unit/test_models.py`

- [ ] **Step 1: Create dependencies and an isolated environment**

Create `requirements.txt`:

```text
numpy>=1.26,<3
onnxruntime>=1.18,<2
tokenizers>=0.19,<1
```

Create `requirements-dev.txt`:

```text
-r requirements.txt
pytest>=8,<9
pytest-cov>=5,<7
ruff>=0.6,<1
scikit-learn>=1.5,<2
```

Create `requirements-assets.txt`:

```text
-r requirements-dev.txt
optimum[onnxruntime]>=1.21,<3
sentence-transformers>=3,<6
torch>=2.2,<3
```

Run:

```powershell
& "C:\Users\renha\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Expected: `.venv` is created and pytest imports successfully.

- [ ] **Step 2: Write failing shared-model tests**

Create `tests/unit/test_models.py`:

```python
from compasscart.models import Constraint, SessionState


def test_session_state_returns_only_current_active_constraints():
    state = SessionState(session_id="s1")
    state.constraints.extend([
        Constraint("color", "red", 1.0, True, "message", 1, 1, "superseded"),
        Constraint("color", "blue", 1.0, True, "message", 3, 2, "active"),
    ])
    assert [(item.attribute, item.value) for item in state.active_constraints()] == [("color", "blue")]
```

- [ ] **Step 3: Run the test and verify collection fails**

Run:

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_models.py -v
```

Expected: FAIL because `compasscart.models` does not exist.

- [ ] **Step 4: Add package and shared models**

Create `src/compasscart/models.py` with immutable `Constraint`, mutable `SessionState`, `RetrievalPlan`, `Candidate`, and `QuestionDecision` dataclasses matching the approved design. `RetrievalPlan` filter mappings use empty `default_factory=dict` values so lexical-only callers can supply only route and query text. `SessionState.active_constraints()` must filter `status == "active"` and the current intent version, while allowing retained category constraints from earlier versions. Because `Constraint` is frozen, Ledger updates use `dataclasses.replace()` instead of mutation.

Create `src/compasscart/config.py` with a frozen `RuntimeConfig` containing `candidate_limit=500`, `rrf_k=60`, `max_recommendations=10`, `component_timeout_ms=800`, dense asset paths, BM25 field weights, and route weights from the spec.

- [ ] **Step 5: Run tests**

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_models.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add requirements.txt requirements-dev.txt requirements-assets.txt src tests/unit/test_models.py
git commit -m "feat: define CompassCart runtime contracts"
```

## Task 3: Normalize Catalog Fields and Build Lexical Index

**Files:**
- Create: `src/compasscart/normalization.py`
- Create: `src/compasscart/catalog.py`
- Create: `tests/fixtures/catalog.jsonl`
- Create: `tests/unit/test_normalization.py`
- Create: `tests/unit/test_catalog.py`

- [ ] **Step 1: Create a four-product fixture**

The fixture must include a red cotton dress, blue running shoes, black leather belt, and waterproof winter jacket with distinct `parent_asin`, price, categories, features, details, rating, and store fields.

- [ ] **Step 2: Write failing normalization tests**

```python
from compasscart.normalization import extract_attributes, terms


def test_extracts_material_color_budget_and_use_case():
    product = {
        "title": "Blue mesh running shoes",
        "price": 79.99,
        "features": ["Breathable mesh for gym and outdoor running"],
        "details": {"Color": "Navy Blue", "Material": "Mesh"},
    }
    attrs = extract_attributes(product)
    assert "blue" in attrs["color"]
    assert "mesh" in attrs["material"]
    assert "running" in attrs["use_case"]
    assert attrs["budget"] == ("79.99",)


def test_terms_are_normalized_and_deduplicated():
    assert terms("Blue BLUE running-shoes") == ["blue", "running", "shoes"]
```

- [ ] **Step 3: Implement deterministic normalization**

`normalization.py` must expose `flatten_text`, `terms`, `normalize_value`, `extract_attributes`, and `searchable_fields`. Use bounded regex and explicit vocabularies; never call external services.

- [ ] **Step 4: Write failing catalog tests**

```python
from compasscart.catalog import CatalogIndex
from compasscart.models import RetrievalPlan


def test_catalog_returns_valid_unique_matches(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    plan = RetrievalPlan(route="buying", query_text="blue running shoes")
    matches = index.search_lexical(plan, limit=10)
    assert matches[0].parent_asin == "SHOE1"
    assert len({item.parent_asin for item in matches}) == len(matches)


def test_attribute_lookup_filters_material(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    assert index.attribute_ids("material", "leather") == {"BELT1"}
```

- [ ] **Step 5: Implement `CatalogIndex`**

Build an in-memory SQLite FTS5 table using the official starter's field order and approved weights. Also build `products`, `valid_ids`, `attributes`, `attribute_inverted`, `category_ids`, and deterministic quality priors. `search_lexical()` returns normalized `Candidate` objects and catches FTS errors by using a pure-Python token-overlap fallback.

- [ ] **Step 6: Run focused tests**

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_normalization.py tests/unit/test_catalog.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/compasscart/normalization.py src/compasscart/catalog.py tests/fixtures tests/unit
git commit -m "feat: build normalized catalog search index"
```

## Task 4: Implement Versioned Dialog State and Parsing

**Files:**
- Create: `src/compasscart/parser.py`
- Create: `src/compasscart/state.py`
- Create: `tests/unit/test_parser.py`
- Create: `tests/unit/test_state.py`

- [ ] **Step 1: Write failing parser tests**

Cover explicit color/material/budget extraction, vague Browsing messages, Buying hard constraints, `Actually, ignore my earlier preference` override, and no-preference replies.

```python
def test_parser_marks_override_and_new_constraint(parser):
    result = parser.parse("Actually, ignore red. What I need is blue leather.", turn=3)
    assert result.is_override is True
    assert {(x.attribute, x.value) for x in result.constraints} >= {
        ("color", "blue"), ("material", "leather")
    }
```

- [ ] **Step 2: Implement `MessageParser`**

Return a `ParseResult` containing route hints, normalized constraints, `is_override`, and `no_preference_attribute`. Explicit message constraints use confidence `1.0`; inferred use `0.6`; profile parsing is handled separately at reset with confidence `0.25`.

- [ ] **Step 3: Write failing Ledger tests**

```python
def test_override_supersedes_conflicting_old_constraint(store):
    store.reset("s1", {"preference_tags": []})
    store.update("s1", "I need a red cotton dress", 1)
    state = store.update("s1", "Actually, ignore red. I need blue.", 3)
    active = {(c.attribute, c.value) for c in state.active_constraints()}
    assert ("color", "blue") in active
    assert ("color", "red") not in active
    assert any(c.value == "red" and c.status == "superseded" for c in state.constraints)
```

- [ ] **Step 4: Implement `SessionStore`**

Use a bounded dictionary keyed by session ID. `reset()` creates version 1 and profile soft constraints. `update()` is idempotent for the same turn/message pair, increments version on override, merges identical values, rejects profile conflicts, records no-preference attributes, and never mutates states from other sessions.

- [ ] **Step 5: Run focused tests and commit**

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_parser.py tests/unit/test_state.py -v
git add src/compasscart/parser.py src/compasscart/state.py tests/unit
git commit -m "feat: track versioned shopping intent"
```

Expected: tests PASS before commit.

## Task 5: Add Routing, Hybrid Retrieval, and Constraint Ranking

**Files:**
- Create: `src/compasscart/router.py`
- Create: `src/compasscart/retrieval.py`
- Create: `src/compasscart/ranker.py`
- Create: `tests/unit/test_router.py`
- Create: `tests/unit/test_retrieval.py`
- Create: `tests/unit/test_ranker.py`

- [ ] **Step 1: Write route tests**

Assert that a concrete hard requirement routes to Buying, vague exploration routes to Browsing, and an override with invalidated filters produces the approved balanced route weights.

- [ ] **Step 2: Implement `RoutePlanner`**

Compute specificity from active hard constraints, explicit price/size/material fields, query term count, and candidate-count estimate. Return a new immutable `RetrievalPlan`; do not mutate SessionState.

- [ ] **Step 3: Write RRF tests**

```python
def test_rrf_fuses_and_deduplicates_candidates():
    fused = reciprocal_rank_fusion(
        {"lexical": ["A", "B"], "attribute": ["B", "C"]},
        weights={"lexical": 0.35, "attribute": 0.45},
        k=60,
    )
    assert set(fused) == {"A", "B", "C"}
    assert fused[0] == "B"
```

- [ ] **Step 4: Implement `HybridRetriever`**

Retrieve lexical, attribute/category, profile, and optional dense lists independently. Fuse with route weights and cap the union at 500. Implement the exact four-stage relaxation sequence from the spec and always produce at least 10 catalog-valid candidates when the catalog has at least 10 items.

- [ ] **Step 5: Write ranking tests**

Assert that explicit hard conflicts lose to matching products, profile cannot override message constraints, Buying keeps exact matches, and Browsing MMR avoids ten near-duplicate items without displacing the strongest match.

- [ ] **Step 6: Implement `ConstraintRanker`**

Use the seven approved positive features and `-0.60` explicit conflict penalty. Normalize every component to `[0, 1]`; break ties by parent_asin for deterministic output. Apply MMR only for Browsing with `lambda=0.85`.

- [ ] **Step 7: Run focused tests and commit**

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_router.py tests/unit/test_retrieval.py tests/unit/test_ranker.py -v
git add src/compasscart/router.py src/compasscart/retrieval.py src/compasscart/ranker.py tests/unit
git commit -m "feat: route and rank hybrid product candidates"
```

## Task 6: Implement Conversion-Aware Clarification

**Files:**
- Create: `src/compasscart/question_policy.py`
- Create: `tests/unit/test_question_policy.py`

- [ ] **Step 1: Write failing utility tests**

```python
def test_policy_chooses_attribute_with_largest_expected_top10_gain(policy, candidate_set):
    # Material splits the weighted candidates cleanly; color is mostly missing.
    decision = policy.choose(candidate_set, state_at_turn_2)
    assert decision.ask_attribute == "material"
    assert decision.utility > 0


def test_policy_never_repeats_or_asks_rejected_attribute(policy, candidate_set):
    state = state_with(asked_attributes=["color"], no_preference_attributes={"size"})
    assert policy.choose(candidate_set, state).ask_attribute not in {"color", "size"}


def test_policy_stops_asking_when_candidate_set_fits_top10(policy):
    assert policy.choose(candidate_set[:10], state_at_turn_2).ask_attribute is None
```

- [ ] **Step 2: Implement weighted expected gain**

Normalize candidate rank scores to probabilities. For each allowed attribute, partition candidates by normalized value, estimate current and post-answer Top 10 probability, multiply gain by coverage, response likelihood, and remaining-turn factor, then apply repeat and no-preference penalties.

- [ ] **Step 3: Implement turn and fallback rules**

Enforce: no question at 10 or fewer candidates; one question per attribute; after turn 8 require utility above 0.15; `other` only once when all standard utility values are below 0.03 and candidate count exceeds 200.

- [ ] **Step 4: Run tests and commit**

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_question_policy.py -v
git add src/compasscart/question_policy.py tests/unit/test_question_policy.py
git commit -m "feat: ask conversion-aware clarification questions"
```

## Task 7: Build Contract-Safe Agent Integration

**Files:**
- Create: `src/compasscart/response.py`
- Create: `src/compasscart/tracing.py`
- Create: `src/compasscart/agent.py`
- Create: `agent.py`
- Create: `tools/__init__.py`
- Create: `tools/run_agent.py`
- Create: `tests/contract/test_agent_contract.py`
- Create: `tests/integration/test_scenarios.py`

- [ ] **Step 1: Write contract tests**

Validate exact response keys, allowed `ask_attribute`, non-negative usage, at most ten unique valid IDs, `reset()` requirement, empty-message behavior, and exception containment for turns 1 through 10.

- [ ] **Step 2: Implement `ResponseBuilder`**

Accept ranked candidates and an optional question decision. Remove invalid/duplicate IDs, cap to `top_k`, fill from route-safe fallback, return zero usage tokens, and produce a short attribute-specific question template.

- [ ] **Step 3: Implement bounded tracing**

Use an in-memory deque with maximum 5,000 entries. Store session ID, turn, route, active constraints, component candidate counts, ask_attribute, fallbacks, and component milliseconds. Logging failure disables tracing without affecting responses.

- [ ] **Step 4: Implement orchestrator and official entry**

`src/compasscart/agent.py` owns shared catalog/index objects and per-session state. The exact sequence is parse/update state, build route plan, retrieve, rank, choose the clarification from the current ranked candidates, then build the response. Every component call has a fallback. Root `agent.py` contains only:

```python
from pathlib import Path
import sys

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from compasscart.agent import CompassCartAgent as Agent

__all__ = ["Agent"]
```

- [ ] **Step 5: Add a compatible custom-Agent evaluator runner**

The official evaluator intentionally imports `starter.agent.Agent` and has no `--agent` argument. Preserve it unchanged and create `tools/run_agent.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write four multi-turn scenario tests**

Use deterministic simulated messages for Buying, Browsing, Intent Override, and Boundary. Assert valid recommendations every turn, old constraints disappear after override, rejected attributes are not repeated, and a question never prevents recommendations.

- [ ] **Step 7: Run contract and integration tests**

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/contract tests/integration -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add agent.py src/compasscart tools/__init__.py tools/run_agent.py tests/contract tests/integration
git commit -m "feat: integrate contract-safe CompassCart agent"
```

## Task 8: Add Optional Offline Dense Retrieval

**Files:**
- Create: `src/compasscart/dense.py`
- Create: `tools/build_dense_assets.py`
- Create: `tests/unit/test_dense.py`
- Modify: `src/compasscart/catalog.py`
- Modify: `src/compasscart/retrieval.py`

- [ ] **Step 1: Write dense fallback tests**

Assert that missing assets, missing optional packages, checksum mismatch, and inference exceptions return an unavailable backend without breaking lexical search. A tiny fake embedding backend must prove that dense candidates participate in RRF when available.

- [ ] **Step 2: Implement `DenseBackend` protocol and null backend**

Expose `available: bool` and `search(text, limit) -> list[Candidate]`. `NullDenseBackend` always returns an empty list. No other module imports ONNX Runtime directly.

- [ ] **Step 3: Implement local ONNX backend**

Load tokenizer JSON, quantized ONNX model, product ID array, int8 vectors, scales, and SHA256 manifest from `assets/`. Mean-pool token embeddings with attention masks, L2-normalize, compute cosine scores with NumPy, and use partial top-k selection. Any load error returns the null backend.

- [ ] **Step 4: Implement offline asset builder**

`tools/build_dense_assets.py` must read the frozen catalog, export/quantize `sentence-transformers/all-MiniLM-L6-v2`, encode normalized product text in batches, quantize vectors to int8, write IDs/scales/vectors, and create SHA256SUMS. It may use network and development dependencies; runtime code may not.

Install its isolated build dependencies only when this task starts:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-assets.txt
```

- [ ] **Step 5: Run dense tests and full lexical fallback tests**

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_dense.py tests/unit/test_catalog.py tests/unit/test_retrieval.py -v
```

Expected: PASS with and without optional packages/assets.

- [ ] **Step 6: Commit code, not generated model assets**

```powershell
git add src/compasscart/dense.py src/compasscart/catalog.py src/compasscart/retrieval.py tools/build_dense_assets.py tests/unit/test_dense.py
git commit -m "feat: add offline dense retrieval with lexical fallback"
```

## Task 9: Add Cross-Validation and Failure Analysis

**Files:**
- Create: `tools/run_cv.py`
- Create: `tools/analyze_failures.py`
- Create: `tests/unit/test_experiment_tools.py`
- Create: `reports/experiments/.gitkeep`

- [ ] **Step 1: Write split tests**

Assert deterministic five-fold assignment, preserved scenario proportions, no duplicate sample IDs across folds, and a sealed fold that is excluded from development reports before an explicit `--audit` flag.

- [ ] **Step 2: Implement `run_cv.py`**

Load public JSONL, stratify by `(scenario_type, difficulty_bucket)` with a fixed seed, run official evaluation against folds 1-4, compute mean/std and `selection_score`, and write timestamped JSON containing commit, config hash, aggregate metrics, scenario metrics, latency, fallbacks, and API cost metadata.

- [ ] **Step 3: Implement `analyze_failures.py`**

Read a results JSON and group misses by route, scenario, turn, last candidate count, asked attributes, override state, and fallback. Output a compact Markdown report with counts and representative sample IDs, not hard-coded fixes.

- [ ] **Step 4: Run tests and smoke CV**

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_experiment_tools.py -v
& .\.venv\Scripts\python.exe -m tools.run_cv --folds 1 2 3 4 --agent agent:Agent
```

Expected: deterministic report under `reports/experiments/` and no fold-5 sample details.

- [ ] **Step 5: Commit**

```powershell
git add tools/run_cv.py tools/analyze_failures.py tests/unit/test_experiment_tools.py reports/experiments/.gitkeep
git commit -m "feat: add reproducible scoring and failure analysis"
```

## Task 10: Verify Fault Tolerance and Performance

**Files:**
- Create: `tests/performance/test_runtime.py`
- Create: `tests/integration/test_fallbacks.py`
- Modify: modules only when a failing test exposes a specific defect

- [ ] **Step 1: Write fallback integration tests**

Inject missing dense assets, FTS query errors, parser exceptions, ranker timeouts, missing sessions, empty messages, empty retrieval results, duplicate IDs, and trace write failures. Every call must return a contract-valid response from the fixture catalog.

- [ ] **Step 2: Write bounded-state performance tests**

Run 800 synthetic sessions and assert SessionStore/trace bounds, no unhandled exceptions, no invalid IDs, and fixture P95 under 2 seconds. Mark the real-catalog benchmark separately so normal unit tests stay fast.

- [ ] **Step 3: Run offline and dependency-free matrix**

```powershell
$env:PYTHONPATH = "src;."
$env:COMPASSCART_DISABLE_DENSE = "1"
& .\.venv\Scripts\python.exe -m pytest tests/integration/test_fallbacks.py tests/performance/test_runtime.py -v
Remove-Item Env:COMPASSCART_DISABLE_DENSE
& .\.venv\Scripts\python.exe -m pytest -q
```

Expected: PASS in lexical-only and normal configurations.

- [ ] **Step 4: Run static checks**

```powershell
& .\.venv\Scripts\python.exe -m ruff check agent.py src tools tests
& .\.venv\Scripts\python.exe -m ruff format --check agent.py src tools tests
```

Expected: no diagnostics.

- [ ] **Step 5: Commit fixes and tests**

```powershell
git add src tests
git commit -m "test: harden CompassCart offline execution"
```

## Task 11: Package Submission and Complete Documentation

**Files:**
- Create: `tools/package_submission.py`
- Create: `README.md`
- Create: `reports/final/architecture.md`
- Create: `reports/final/ablation-template.md`
- Create: `reports/final/demo-script.md`
- Create: `tests/contract/test_submission_package.py`

- [ ] **Step 1: Write package tests**

Assert the generated `dist/compasscart-submission.zip` contains `agent.py`, requirements, `src/`, required lightweight assets, README, licenses, and SHA256SUMS; assert it excludes API keys, `.env`, public labels, evaluator, organizer files, caches, traces, and experiment reports.

- [ ] **Step 2: Implement deterministic packager**

`package_submission.py` copies an allowlist into a temporary directory, scans text files for secret patterns, verifies asset checksums, imports `Agent`, runs a fixture response, creates a stable zip, and prints file count, size, and SHA256.

- [ ] **Step 3: Write team README**

Include project overview, exact Python version, installation, catalog path, optional dense assets, one-command evaluator run, offline behavior, architecture, metrics, costs, limitations, licenses, and five-member contributions. Do not copy organizer-only text.

- [ ] **Step 4: Write final evidence templates**

The architecture report maps each module to a judging criterion. The ablation table has rows for starter, lexical, +state, +question policy, +dense, and full system. The demo script shows one Browsing and one Intent Override trajectory, then repeats with network disabled.

- [ ] **Step 5: Run release verification**

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m tools.package_submission
& .\.venv\Scripts\python.exe -m pytest tests/contract/test_submission_package.py -v
git status --short
```

Expected: all tests pass, package checksum prints, and only intentional generated files are untracked/ignored.

- [ ] **Step 6: Commit**

```powershell
git add README.md tools/package_submission.py reports/final tests/contract/test_submission_package.py
git commit -m "docs: prepare reproducible CompassCart submission"
```

## Task 12: Run Official Evaluation and Select Stable Release

**Files:**
- Modify: only configuration values supported by bounded search
- Create: `reports/final/final-results.json`
- Create: `reports/final/release-checklist.md`

- [ ] **Step 1: Run development folds and choose one candidate**

Run folds 1-4 for every bounded configuration, record all reports, and select exactly one release candidate by `selection_score`. Tag it before viewing fold 5.

```powershell
git tag -a compasscart-audit-candidate -m "candidate selected before sealed audit"
```

- [ ] **Step 2: Run the sealed fold once**

```powershell
& .\.venv\Scripts\python.exe -m tools.run_cv --audit --folds 5 --agent agent:Agent
```

Expected: fold-5 TechnicalScore at least 0.35, valid completion 100%, and no scenario score near zero. Do not tune based on the result.

- [ ] **Step 3: Run full official public evaluator**

```powershell
& .\.venv\Scripts\python.exe -m tools.run_agent
```

Expected: a reproducible final results JSON with all official aggregate and scenario metrics.

- [ ] **Step 4: Execute final offline checklist**

Verify no network dependency, no API keys, cold install/run instructions, 800-session stability, asset checksums, package import, license attribution, Devpost text, demo video script, and a rollback tag.

- [ ] **Step 5: Commit evidence and tag release**

```powershell
git add reports/final/final-results.json reports/final/release-checklist.md
git commit -m "chore: record CompassCart release evidence"
git tag -a compasscart-v1 -m "TechJam 2026 Track 4 submission"
```

## Plan Completion Checks

- Every runtime module is reachable from an integration or contract test.
- Every approved fallback has a dedicated failure-injection test.
- The Agent imports and responds with standard-library lexical fallback even when optional dependencies/assets are unavailable.
- Official evaluator inputs, labels, and scoring code remain unchanged.
- Fold 5 is sealed until a single candidate is tagged.
- The final zip is allowlist-based, secret-scanned, checksum-verified, offline-capable, and reproducible.
- Documentation contains actual measured metrics and costs before release; templates never ship as final evidence.
