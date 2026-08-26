# CompassCart Score Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise CompassCart's private-set generalization by making parser, constraint matching, candidate generation, and ranking use consistent product semantics, then accept only changes that improve stable development-fold metrics.

**Architecture:** Preserve the offline Agent contract and frozen dense assets. First remove parser/state defects, then add cached catalog semantics shared by filtering, attribute recall, fallback, and ranking; only after that baseline is measured, run bounded fusion and MMR experiments that can be rejected independently.

**Tech Stack:** Python 3.10+, standard library (`dataclasses`, `json`, `sqlite3`, collections), optional NumPy/ONNX Runtime/tokenizers, pytest, Ruff, and the unchanged official evaluator.

---

## Scope, Baseline, And File Map

The approved design is `docs/superpowers/specs/2026-08-26-compasscart-score-optimization-design.md`.

The fresh Windows baseline is:

- TechnicalScore `0.518309`
- HitRate@10 `0.625`
- MRR `0.321365`
- MTTC `5.530`

The development selection baseline is `0.500756` from folds 1-4. The final accepted version must reach at least `0.510756`, with mean TechnicalScore at least `0.529195`, and satisfy the scenario/fold safeguards in the design.

| File | Responsibility In This Plan |
|---|---|
| `src/compasscart/parser.py` | Prevent cross-attribute aliases, preserve route on no-preference, make unknown clarification values soft |
| `src/compasscart/state.py` | Preserve the existing route and bounded query evidence |
| `src/compasscart/normalization.py` | Canonical category term sets and searchable term sets |
| `src/compasscart/constraints.py` | One category/open-text matching truth table |
| `src/compasscart/catalog.py` | Cache semantic terms and expose semantic category lookup/matching |
| `src/compasscart/retrieval.py` | Reuse catalog semantics for attribute candidates, exact filtering, and fallback |
| `src/compasscart/ranker.py` | Reuse catalog matching and optionally restore bounded fused-score evidence |
| `src/compasscart/config.py` | Hold accepted rank fusion and MMR values |
| `src/compasscart/agent.py` | Pass accepted ranker configuration |
| `tools/compare_results.py` | Produce reproducible baseline/candidate session differences |
| `tests/unit/test_parser.py` | Parser disambiguation regressions |
| `tests/unit/test_state.py` | Route and unknown-clarification regressions |
| `tests/unit/test_normalization.py` | Category singularization and term-set regressions |
| `tests/unit/test_constraints.py` | Shared matching regressions |
| `tests/unit/test_catalog.py` | Cache and semantic category lookup regressions |
| `tests/unit/test_retrieval.py` | Filtering/candidate/fallback consistency regressions |
| `tests/unit/test_ranker.py` | Catalog matching and bounded fusion regressions |
| `tests/unit/test_experiment_tools.py` | Session comparison regressions |

Do not modify `evaluator/`, `data/public_set.jsonl`, `data/catalog.jsonl`, any ground-truth field, or dense assets. Do not add sample IDs, ASINs, or public answers to production code.

## Shared Evaluation Commands

Use these PowerShell variables for every task:

```powershell
$env:PYTHONPATH = "src;."
$python = ".\.venv\Scripts\python.exe"
$ruff = ".\.venv\Scripts\ruff.exe"
```

Run development CV with the unchanged seed and folds:

```powershell
& $python -m tools.run_cv `
  --catalog data/catalog.jsonl `
  --dataset data/public_set.jsonl `
  --folds 1 2 3 4 `
  --seed 2026 `
  --output-dir var/score-optimization/cv
```

Never use fold 5 to select a change. Record the four fold scores, mean, standard deviation, selection score, per-scenario metrics, and latency before deciding whether to keep a stage.

### Task 1: Fix Parser Disambiguation And Route Preservation

**Files:**
- Modify: `src/compasscart/parser.py:255-530`
- Modify: `tests/unit/test_parser.py`
- Modify: `tests/unit/test_state.py`

- [ ] **Step 1: Write the failing no-preference and clarification tests**

Add this assertion to `test_parser_detects_no_preference_and_browsing_route` in `tests/unit/test_parser.py`:

```python
assert no_preference.route_hint is None
```

Add these tests to `tests/unit/test_state.py`:

```python
def test_no_preference_keeps_existing_buying_route():
    store = SessionStore(MessageParser())
    store.reset("s1", {})
    store.update("s1", "I need blue shoes", 1)

    state = store.update(
        "s1", "No preference.", 2, expected_attribute="size"
    )

    assert state.route == "buying"
    assert state.route_hint == "buying"


def test_unrecognized_clarification_stays_query_evidence_without_hard_constraint():
    store = SessionStore(MessageParser({"feature": ("Waterproof",)}))
    store.reset("s1", {})

    state = store.update(
        "s1",
        "with a magnetic clasp",
        2,
        expected_attribute="feature",
    )

    assert state.query_history == ["with a magnetic clasp"]
    assert Agent._query_text("with a magnetic clasp", state).count(
        "with a magnetic clasp"
    ) == 1
    clarification = [
        item for item in state.active_constraints() if item.source == "clarification"
    ]
    assert len(clarification) == 1
    assert clarification[0].is_hard is False
    assert clarification[0].confidence == 0.6
```

- [ ] **Step 2: Write the failing protected-category-span tests**

Add to `tests/unit/test_parser.py`:

```python
def test_category_span_blocks_nested_dynamic_style_and_brand_aliases():
    parser = MessageParser(
        {
            "category": ("Fashion Sneakers", "Boy Shorts"),
            "style": ("Sneaker",),
            "brand": ("Boy",),
        }
    )

    sneakers = parser.parse("I want Fashion Sneakers", turn=1)
    shorts = parser.parse("I want Boy Shorts", turn=1)

    assert ("category", "fashion sneakers") in {
        (item.attribute, item.value) for item in sneakers.constraints
    }
    assert ("category", "boy shorts") in {
        (item.attribute, item.value) for item in shorts.constraints
    }
    assert not any(
        item.attribute == "style" and item.is_hard
        for item in sneakers.constraints
    )
    assert not any(
        item.attribute == "brand" and item.is_hard
        for item in shorts.constraints
    )


def test_explicit_style_and_brand_cues_survive_category_span_protection():
    parser = MessageParser(
        {
            "category": ("Fashion Sneakers", "Boy Shorts"),
            "style": ("Sneaker",),
            "brand": ("Boy",),
        }
    )

    styled = parser.parse("I want sneaker style Fashion Sneakers", turn=1)
    branded = parser.parse("I want Boy brand shorts", turn=1)

    assert ("style", "sneaker") in {
        (item.attribute, item.value) for item in styled.constraints
    }
    assert ("brand", "boy") in {
        (item.attribute, item.value) for item in branded.constraints
    }


def test_generic_taxonomy_phrase_does_not_become_a_specific_hard_category():
    parser = MessageParser(
        {"category": ("Shoes", "Jewelry", "Fashion Sneakers")}
    )

    result = parser.parse(
        "I'm looking for Shoes & Jewelry Men, but I'm still exploring.",
        turn=1,
    )

    assert not any(
        item.attribute == "category" and item.is_hard
        for item in result.constraints
    )
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
& $python -m pytest `
  tests/unit/test_parser.py `
  tests/unit/test_state.py `
  tests/unit/test_router.py -q
```

Expected: failures show that no-preference returns `route_hint="browsing"`, unknown clarification is hard with confidence `1.0`, and category spans emit nested style/brand constraints. Existing unrelated tests must remain green.

- [ ] **Step 4: Implement the minimal parser changes**

In `MessageParser.parse`, replace the no-preference return with:

```python
if _NO_PREFERENCE_RE.search(text):
    attribute = expected_attribute or self._mentioned_attribute(text)
    return ParseResult(
        route_hint=None,
        no_preference_attribute=attribute,
        is_override=is_override,
        is_continuation=is_continuation,
        replace_preferences=replace_preferences,
    )
```

In `_extract_expected`, return soft clarification evidence:

```python
if cleaned and attribute in {
    "brand",
    "category",
    "feature",
    "other",
    "size",
    "style",
    "use_case",
}:
    return [ParsedConstraint(attribute, cleaned, 0.6, False, source)]
return []
```

Change the `_alias_allowed` call to pass the matched end offset:

```python
if not self._alias_allowed(attribute, value, text, start, end):
    continue
```

Change its signature accordingly:

```python
def _alias_allowed(
    self, attribute: str, value: str, text: str, start: int, end: int
) -> bool:
```

Add these helpers to `MessageParser`:

```python
@staticmethod
def _spans_overlap(
    left: tuple[int, int], right: tuple[int, int]
) -> bool:
    return left[0] < right[1] and right[0] < left[1]

@staticmethod
def _has_explicit_attribute_cue(
    attribute: str, text: str, start: int, end: int
) -> bool:
    before = text[max(0, start - 32) : start]
    after = text[end : min(len(text), end + 24)]
    if attribute == "brand":
        return bool(
            re.search(r"\b(?:brand|by|from|made\s+by)\s*$", before)
            or re.search(r"^\s*brand\b", after)
            or re.search(r"\bbrand\b", text[start:end])
            or "'s" in text[start:end]
        )
    if attribute == "style":
        return bool(
            re.search(r"\b(?:style|look|design)\s*$", before)
            or re.search(r"^\s*(?:style|look|design)\b", after)
        )
    return False
```

Add this module-level pattern beside `_CATEGORY_RE`:

```python
_GENERIC_TAXONOMY_RE = re.compile(
    r"\b(?:clothing\s*,?\s*)?shoes?\s*(?:&|and)\s+jewelry"
    r"(?:\s+(?:men|women))?\b",
    re.IGNORECASE,
)
```

Immediately after the existing fixed-value early return in `_alias_allowed`, require a cue for catalog-derived style/brand aliases:

```python
if attribute in {"brand", "style"} and not self._has_explicit_attribute_cue(
    attribute, text, start, end
):
    return False
```

After `candidates` is collected in `_vocabulary_matches` and before it is sorted, filter only dynamic nested aliases:

```python
generic_taxonomy_spans = [
    match.span() for match in _GENERIC_TAXONOMY_RE.finditer(text)
]
candidates = [
    candidate
    for candidate in candidates
    if not (
        candidate[2] == "category"
        and any(
            self._spans_overlap((candidate[0], candidate[1]), taxonomy_span)
            for taxonomy_span in generic_taxonomy_spans
        )
    )
]
category_spans = [
    (start, end)
    for start, end, attribute, _ in candidates
    if attribute == "category"
]
candidates = [
    candidate
    for candidate in candidates
    if not (
        candidate[2] in {"brand", "style"}
        and normalize_value(candidate[3])
        not in self._fixed_values.get(candidate[2], set())
        and any(
            self._spans_overlap((candidate[0], candidate[1]), category_span)
            for category_span in category_spans
        )
        and not self._has_explicit_attribute_cue(
            candidate[2], text, candidate[0], candidate[1]
        )
    )
]
```

Keep the existing route persistence in `SessionStore.update`; it already changes state only when `result.route_hint is not None`.

- [ ] **Step 5: Verify GREEN and run the parser/state regression set**

Run:

```powershell
& $python -m pytest `
  tests/unit/test_parser.py `
  tests/unit/test_state.py `
  tests/unit/test_router.py `
  tests/integration/test_scenarios.py `
  tests/integration/test_functional_edge_cases.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Run S1 development CV and apply the gate**

Run the shared folds 1-4 command. Keep S1 if selection score does not fall by more than `0.005`, no fold falls by more than `0.02`, and the two semantic defects are fixed. Otherwise inspect regressions and revise only the parser rules before continuing.

- [ ] **Step 7: Commit S1**

```powershell
git add src/compasscart/parser.py tests/unit/test_parser.py tests/unit/test_state.py
git commit -m "fix: preserve intent while parsing catalog language"
```

### Task 2: Add Shared Category And Open-Text Semantics

**Files:**
- Modify: `src/compasscart/normalization.py:94-199`
- Modify: `src/compasscart/constraints.py`
- Modify: `src/compasscart/catalog.py:22-137`
- Modify: `tests/unit/test_normalization.py`
- Modify: `tests/unit/test_constraints.py`
- Modify: `tests/unit/test_catalog.py`

- [ ] **Step 1: Write the failing normalization and matching tests**

Add to `tests/unit/test_normalization.py`:

```python
from compasscart.normalization import category_term_set, normalize_category_value


def test_category_terms_handle_taxonomy_union_and_clothing_plural_exceptions():
    assert category_term_set(("Shoes", "Athletic")) == frozenset(
        {"shoe", "athletic"}
    )
    assert normalize_category_value("hoodies") == "hoodie"
    assert normalize_category_value("booties") == "bootie"
```

Add to `tests/unit/test_constraints.py`:

```python
def test_category_matching_uses_the_union_of_taxonomy_values():
    constraint = _constraint("category", "athletic shoes")

    assert matches_constraint(
        {}, {"category": ("Shoes", "Athletic")}, constraint
    )
    assert not matches_constraint(
        {}, {"category": ("Shoes", "Formal")}, constraint
    )


def test_soft_clarification_matches_searchable_product_text_only():
    product = {
        "features": ["Machine washable with reinforced seams"],
        "description": ["A durable everyday layer"],
    }
    clarification = Constraint(
        "feature",
        "machine washable with reinforced seams",
        0.6,
        False,
        "clarification",
        2,
        1,
    )
    same_message = Constraint(
        "feature",
        "machine washable with reinforced seams",
        1.0,
        True,
        "message",
        1,
        1,
    )

    assert matches_constraint(product, {"feature": ()}, clarification)
    assert not matches_constraint(product, {"feature": ()}, same_message)
```

- [ ] **Step 2: Write the failing catalog cache tests**

Update imports in `tests/unit/test_catalog.py` and add:

```python
from compasscart.normalization import category_term_set, searchable_term_set


@pytest.mark.parametrize("enable_fts", [True, False])
def test_catalog_caches_category_and_searchable_terms(
    fixture_catalog_path, enable_fts
):
    index = CatalogIndex(fixture_catalog_path, enable_fts=enable_fts)
    product = index.product("SHOE1")

    assert index.category_terms["SHOE1"] == frozenset({"shoe", "athletic"})
    assert index.searchable_terms["SHOE1"] == searchable_term_set(product)
    assert index.category_term_inverted["shoe"] == {"SHOE1"}
    assert index.category_ids("athletic shoes") == {"SHOE1"}
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
& $python -m pytest `
  tests/unit/test_normalization.py `
  tests/unit/test_constraints.py `
  tests/unit/test_catalog.py -q
```

Expected: import or assertion failures for missing term-set functions, exact category equality, open clarification matching, and missing catalog caches.

- [ ] **Step 4: Implement normalized term sets**

Add near the normalization functions in `src/compasscart/normalization.py`:

```python
_CATEGORY_TERM_EXCEPTIONS = {
    "booties": "bootie",
    "hoodies": "hoodie",
    "panties": "panty",
    "ties": "tie",
}
_CATEGORY_LINK_TERMS = {"and", "for", "of"}


def category_term_set(value: object) -> frozenset[str]:
    return frozenset(
        _singular_category_term(token)
        for token in terms(value)
        if token not in _CATEGORY_LINK_TERMS
    )


def searchable_term_set(product: dict[str, object]) -> frozenset[str]:
    return frozenset(terms(searchable_fields(product)))
```

Replace `_singular_category_term` with:

```python
def _singular_category_term(token: str) -> str:
    if token in _CATEGORY_TERM_EXCEPTIONS:
        return _CATEGORY_TERM_EXCEPTIONS[token]
    if len(token) > 3 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 2 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
```

- [ ] **Step 5: Implement category and open-clarification matching**

Import `Collection` and the new normalization functions in `constraints.py`. Extend `matches_constraint` with keyword-only caches and use this complete non-budget branch:

```python
def matches_constraint(
    product: dict[str, object],
    attributes: dict[str, tuple[str, ...]],
    constraint: Constraint,
    *,
    category_terms: Collection[str] | None = None,
    searchable_terms: Collection[str] | None = None,
) -> bool:
    if constraint.operator in {"lte", "gte", "between"}:
        return constraint.attribute == "budget" and _matches_budget_range(
            product, constraint
        )
    if constraint.attribute == "budget":
        return _matches_budget_equality(product, constraint)

    if constraint.attribute == "category":
        available = set(
            category_terms
            if category_terms is not None
            else category_term_set(attributes.get("category", ()))
        )
        if not available:
            return False
        alternatives = [category_term_set(value) for value in constraint.values()]
        matched = any(wanted and wanted <= available for wanted in alternatives)
    elif constraint.source == "clarification" and not constraint.is_hard:
        available = set(
            searchable_terms
            if searchable_terms is not None
            else searchable_term_set(product)
        )
        alternatives = [frozenset(terms(value)) for value in constraint.values()]
        matched = any(wanted and wanted <= available for wanted in alternatives)
    else:
        values = _product_values(attributes, constraint.attribute)
        if not values:
            return False
        wanted = {normalize_value(value) for value in constraint.values()}
        normalized_values = {normalize_value(value) for value in values}
        matched = bool(normalized_values & wanted)

    if constraint.operator in {"eq", "in"}:
        return matched
    if constraint.operator == "not_in":
        return not matched
    return False
```

Replace `hard_constraint_violations` with:

```python
def hard_constraint_violations(
    product: dict[str, object],
    attributes: dict[str, tuple[str, ...]],
    constraints: Iterable[Constraint],
    *,
    category_terms: Collection[str] | None = None,
    searchable_terms: Collection[str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        display_constraint(constraint)
        for constraint in constraints
        if constraint.is_hard
        and not matches_constraint(
            product,
            attributes,
            constraint,
            category_terms=category_terms,
            searchable_terms=searchable_terms,
        )
    )
```

- [ ] **Step 6: Build catalog caches and semantic category lookup**

In `CatalogIndex.__init__`, add:

```python
self.category_terms: dict[str, frozenset[str]] = {}
self.searchable_terms: dict[str, frozenset[str]] = {}
self.category_term_inverted: dict[str, set[str]] = defaultdict(set)
```

During `_load`, immediately after `attributes = extract_attributes(product)`, add:

```python
category_tokens = category_term_set(attributes.get("category", ()))
search_tokens = searchable_term_set(product)
```

After storing `self.attributes[parent_asin]`, add:

```python
self.category_terms[parent_asin] = category_tokens
self.searchable_terms[parent_asin] = search_tokens
for token in category_tokens:
    self.category_term_inverted[token].add(parent_asin)
```

Add these methods to `CatalogIndex`:

```python
def category_ids(self, value: object) -> set[str]:
    wanted = category_term_set(value)
    if not wanted:
        return set()
    groups = [self.category_term_inverted.get(token, set()) for token in wanted]
    if any(not group for group in groups):
        return set()
    return set.intersection(*(set(group) for group in groups))

def matches(self, parent_asin: str, constraint: Constraint) -> bool:
    return matches_constraint(
        self.products[parent_asin],
        self.attributes[parent_asin],
        constraint,
        category_terms=self.category_terms[parent_asin],
        searchable_terms=self.searchable_terms[parent_asin],
    )

def violations(
    self, parent_asin: str, constraints: Iterable[Constraint]
) -> tuple[str, ...]:
    return hard_constraint_violations(
        self.products[parent_asin],
        self.attributes[parent_asin],
        constraints,
        category_terms=self.category_terms[parent_asin],
        searchable_terms=self.searchable_terms[parent_asin],
    )
```

Import `Iterable`, `Constraint`, `category_term_set`, `searchable_term_set`, and `matches_constraint`. Change `attribute_ids` so category queries share the new postings:

```python
def attribute_ids(self, attribute: str, value: str) -> set[str]:
    if attribute == "category":
        return self.category_ids(value)
    return set(
        self.attribute_inverted.get(attribute, {}).get(
            normalize_value(value), set()
        )
    )
```

- [ ] **Step 7: Verify GREEN and run semantics regressions**

Run:

```powershell
& $python -m pytest `
  tests/unit/test_normalization.py `
  tests/unit/test_constraints.py `
  tests/unit/test_catalog.py `
  tests/unit/test_parser.py `
  tests/unit/test_state.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit the shared semantics layer**

```powershell
git add `
  src/compasscart/normalization.py `
  src/compasscart/constraints.py `
  src/compasscart/catalog.py `
  tests/unit/test_normalization.py `
  tests/unit/test_constraints.py `
  tests/unit/test_catalog.py
git commit -m "feat: unify catalog constraint semantics"
```

### Task 3: Make Retrieval And Ranking Reuse Catalog Semantics

**Files:**
- Modify: `src/compasscart/catalog.py:137-205`
- Modify: `src/compasscart/retrieval.py:140-300`
- Modify: `src/compasscart/ranker.py:75-90`
- Modify: `tests/unit/test_retrieval.py`
- Modify: `tests/unit/test_ranker.py`

- [ ] **Step 1: Write the failing retrieval consistency test**

Add to `tests/unit/test_retrieval.py`:

```python
def test_category_filter_candidates_and_fallback_share_semantics(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index)
    category = _hard_constraint("category", "athletic shoes")
    plan = RetrievalPlan(
        route="buying",
        query_text="athletic shoes",
        hard_filters={"category": ("athletic shoes",)},
        hard_constraints=(category,),
    )
    monkeypatch.setattr(index, "popular_ids", lambda _limit: ["DRESS1", "SHOE1"])

    assert retriever._exact_ids(["SHOE1", "DRESS1"], plan) == ["SHOE1"]
    assert retriever._attribute_candidates(plan, 10) == ["SHOE1"]
    assert retriever._fallback_ids(plan)[0] == "SHOE1"
```

Add to `tests/unit/test_ranker.py`:

```python
def test_ranker_uses_catalog_category_union_semantics(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    state = SessionState(
        "s1",
        route="buying",
        constraints=[
            Constraint("category", "athletic shoes", 1.0, True, "message", 1, 1)
        ],
    )

    ranked = ConstraintRanker(index).rank(
        _candidates(index, ["DRESS1", "SHOE1"]), state
    )

    assert ranked[0].parent_asin == "SHOE1"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
& $python -m pytest tests/unit/test_retrieval.py tests/unit/test_ranker.py -q
```

Expected: exact filtering may pass through the Task 2 matcher, while attribute lookup, fallback, or ranker still fails because it retains older semantics.

- [ ] **Step 3: Delegate all hard checks to CatalogIndex**

In `CatalogIndex._matches_hard`, replace the explicit call with:

```python
constraints = plan.effective_hard_constraints()
if constraints:
    return not self.violations(parent_asin, constraints)
```

In `HybridRetriever._violations`, replace the body with:

```python
constraints = plan.effective_hard_constraints()
if not constraints:
    return ()
return self.catalog.violations(identifier, constraints)
```

In `ConstraintRanker._matches`, replace the body with:

```python
return self.catalog.matches(identifier, constraint)
```

- [ ] **Step 4: Use semantic category IDs in candidate generation and fallback**

At the start of `HybridRetriever._ids_for_constraint`, after budget handling, add:

```python
if constraint.attribute == "category":
    matched = set().union(
        *(self.catalog.category_ids(value) for value in constraint.values())
    )
    if constraint.operator == "not_in":
        present = set().union(*self.catalog.attribute_inverted["category"].values())
        return present - matched
    if constraint.operator in {"eq", "in"}:
        return matched
    return set()
```

Replace `_fallback_ids` with:

```python
def _fallback_ids(self, plan: RetrievalPlan) -> list[str]:
    category_groups = [
        self._ids_for_constraint(constraint)
        for constraint in plan.effective_hard_constraints()
        if constraint.attribute == "category"
        and constraint.operator in {"eq", "in"}
    ]
    category_ids = (
        set.intersection(*category_groups) if category_groups else set()
    )
    ordered_category = sorted(
        category_ids, key=lambda item: (-self.catalog.quality[item], item)
    )
    return ordered_category + self.catalog.popular_ids(plan.candidate_limit)
```

- [ ] **Step 5: Verify GREEN and run all core search tests**

Run:

```powershell
& $python -m pytest `
  tests/unit/test_catalog.py `
  tests/unit/test_constraints.py `
  tests/unit/test_retrieval.py `
  tests/unit/test_ranker.py `
  tests/integration/test_scenarios.py `
  tests/integration/test_functional_edge_cases.py `
  tests/integration/test_fallbacks.py -q
```

Expected: all selected tests pass and exact candidates still precede relaxed candidates.

- [ ] **Step 6: Run S2 development CV and apply the final semantic gate**

Run the shared folds 1-4 command. Keep S2 only if it improves S1 selection score, no fold falls more than `0.02` below its S0 counterpart, and combined Buying/Browsing/Intent Override HitRate does not fall more than `0.025` in any scenario. Inspect every Boundary net hit loss.

- [ ] **Step 7: Commit retrieval integration**

```powershell
git add `
  src/compasscart/catalog.py `
  src/compasscart/retrieval.py `
  src/compasscart/ranker.py `
  tests/unit/test_retrieval.py `
  tests/unit/test_ranker.py
git commit -m "fix: share semantics across recall and ranking"
```

### Task 4: Add A Reproducible Result Comparator

**Files:**
- Create: `tools/compare_results.py`
- Modify: `tests/unit/test_experiment_tools.py`

- [ ] **Step 1: Write the failing session-diff test**

Add this import and test to `tests/unit/test_experiment_tools.py`:

```python
from tools.compare_results import compare_results


def test_compare_results_reports_recoveries_regressions_and_metric_deltas():
    baseline = {
        "recommended_technical_score": 0.5,
        "hit_rate_at_10": 0.5,
        "mrr": 0.2,
        "mttc": 6.0,
        "sessions": [
            {"sample_id": "a", "scenario_type": "buying", "hit": False,
             "first_hit_turn": None, "best_rank": None},
            {"sample_id": "b", "scenario_type": "browsing", "hit": True,
             "first_hit_turn": 1, "best_rank": 1},
            {"sample_id": "c", "scenario_type": "buying", "hit": True,
             "first_hit_turn": 3, "best_rank": 5},
        ],
    }
    candidate = {
        "recommended_technical_score": 0.6,
        "hit_rate_at_10": 2 / 3,
        "mrr": 0.3,
        "mttc": 5.0,
        "sessions": [
            {"sample_id": "a", "scenario_type": "buying", "hit": True,
             "first_hit_turn": 2, "best_rank": 3},
            {"sample_id": "b", "scenario_type": "browsing", "hit": False,
             "first_hit_turn": None, "best_rank": None},
            {"sample_id": "c", "scenario_type": "buying", "hit": True,
             "first_hit_turn": 2, "best_rank": 2},
        ],
    }

    report = compare_results(baseline, candidate)

    assert report["metric_delta"]["recommended_technical_score"] == 0.1
    assert report["recovered"] == ["a"]
    assert report["regressed"] == ["b"]
    assert report["rank_or_turn_improved"] == ["c"]
    assert report["by_scenario"]["buying"]["recovered"] == 1
    assert report["by_scenario"]["browsing"]["regressed"] == 1
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
& $python -m pytest tests/unit/test_experiment_tools.py -q
```

Expected: collection fails because `tools.compare_results` does not exist.

- [ ] **Step 3: Implement the comparator**

Create `tools/compare_results.py`:

```python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

_METRICS = (
    "recommended_technical_score",
    "hit_rate_at_10",
    "mrr",
    "mttc",
)


def compare_results(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    baseline_sessions = {
        str(item["sample_id"]): item for item in baseline.get("sessions", [])
    }
    candidate_sessions = {
        str(item["sample_id"]): item for item in candidate.get("sessions", [])
    }
    if set(baseline_sessions) != set(candidate_sessions):
        raise ValueError("baseline and candidate sample IDs differ")

    recovered: list[str] = []
    regressed: list[str] = []
    improved: list[str] = []
    worsened: list[str] = []
    scenario_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "recovered": 0,
            "regressed": 0,
            "rank_or_turn_improved": 0,
            "rank_or_turn_worsened": 0,
        }
    )
    for sample_id in sorted(baseline_sessions):
        before = baseline_sessions[sample_id]
        after = candidate_sessions[sample_id]
        scenario = str(after.get("scenario_type", "unknown"))
        before_hit = bool(before.get("hit"))
        after_hit = bool(after.get("hit"))
        if not before_hit and after_hit:
            recovered.append(sample_id)
            scenario_counts[scenario]["recovered"] += 1
        elif before_hit and not after_hit:
            regressed.append(sample_id)
            scenario_counts[scenario]["regressed"] += 1
        elif before_hit and after_hit:
            before_key = (
                int(before.get("first_hit_turn") or 11),
                int(before.get("best_rank") or 11),
            )
            after_key = (
                int(after.get("first_hit_turn") or 11),
                int(after.get("best_rank") or 11),
            )
            if after_key < before_key:
                improved.append(sample_id)
                scenario_counts[scenario]["rank_or_turn_improved"] += 1
            elif after_key > before_key:
                worsened.append(sample_id)
                scenario_counts[scenario]["rank_or_turn_worsened"] += 1

    metric_delta = {
        metric: round(float(candidate[metric]) - float(baseline[metric]), 6)
        for metric in _METRICS
    }
    return {
        "metric_delta": metric_delta,
        "recovered": recovered,
        "regressed": regressed,
        "rank_or_turn_improved": improved,
        "rank_or_turn_worsened": worsened,
        "by_scenario": dict(sorted(scenario_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare_results(baseline, candidate)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
& $python -m pytest tests/unit/test_experiment_tools.py -q
& $ruff check tools/compare_results.py tests/unit/test_experiment_tools.py
```

Expected: all tests pass and Ruff exits zero.

- [ ] **Step 5: Commit the comparator**

```powershell
git add tools/compare_results.py tests/unit/test_experiment_tools.py
git commit -m "test: compare evaluator result changes"
```

### Task 5: Run A Bounded Fusion-Ranking Experiment

**Files:**
- Modify: `src/compasscart/config.py`
- Modify: `src/compasscart/agent.py:23-48`
- Modify: `src/compasscart/ranker.py:9-73`
- Modify: `tests/unit/test_ranker.py`

- [ ] **Step 1: Write the failing fusion test**

Add to `tests/unit/test_ranker.py`:

```python
def test_fusion_reorders_exact_candidates_without_promoting_relaxed(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    index.quality["DRESS1"] = 0.0
    index.quality["SHOE1"] = 0.0
    index.quality["JACKET1"] = 0.0
    candidates = [
        Candidate(
            "DRESS1",
            index.product("DRESS1"),
            {"lexical": 0.08, "attribute": 0.8},
            0.9,
        ),
        Candidate(
            "SHOE1", index.product("SHOE1"), {"lexical": 0.1}, 0.1
        ),
        Candidate(
            "JACKET1",
            index.product("JACKET1"),
            {"lexical": 0.1, "attribute": 0.9},
            1.0,
            relaxed=True,
        ),
    ]
    state = SessionState("s1", route="buying")

    baseline = ConstraintRanker(index, fusion_weight=0.0).rank(candidates, state)
    experiment = ConstraintRanker(index, fusion_weight=0.10).rank(candidates, state)

    assert [item.parent_asin for item in baseline] == [
        "SHOE1", "DRESS1", "JACKET1"
    ]
    assert [item.parent_asin for item in experiment] == [
        "DRESS1", "SHOE1", "JACKET1"
    ]
    assert experiment[0].score > experiment[1].score
    assert experiment[-1].relaxed is True
```

The pair of ordering assertions proves that bounded fusion changes exact-candidate order while the relaxed product remains last.

- [ ] **Step 2: Run the ranker test and verify RED**

Run:

```powershell
& $python -m pytest tests/unit/test_ranker.py -q
```

Expected: construction fails because `fusion_weight` is not accepted.

- [ ] **Step 3: Implement bounded fusion weight**

Append to `RuntimeConfig`:

```python
rank_fusion_weight: float = 0.10
mmr_lambda: float = 0.85
```

Pass both values in `CompassCartAgent.__init__`:

```python
self.ranker = ConstraintRanker(
    self.catalog,
    fusion_weight=self.config.rank_fusion_weight,
    mmr_lambda=self.config.mmr_lambda,
)
```

Replace the ranker constructor with:

```python
def __init__(
    self,
    catalog: CatalogIndex,
    *,
    fusion_weight: float = 0.0,
    mmr_lambda: float = 0.85,
) -> None:
    if not 0.0 <= fusion_weight <= 0.10:
        raise ValueError("fusion_weight must be between 0.0 and 0.10")
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be between 0.0 and 1.0")
    self.catalog = catalog
    self.fusion_weight = fusion_weight
    self.mmr_lambda = mmr_lambda
```

At the start of `rank`, calculate:

```python
fusion = self._normalized_scores(candidates)
source_weight = (0.40 - self.fusion_weight) / 2.0
```

Replace the score expression with:

```python
score = (
    0.30 * hard_coverage
    + source_weight * lexical[identifier]
    + source_weight * dense[identifier]
    + self.fusion_weight * fusion[identifier]
    + 0.10 * category_match
    + 0.10 * soft_coverage
    + 0.05 * profile_affinity
    + 0.05 * self.catalog.quality.get(identifier, 0.0)
    - 0.60 * conflict
)
```

Add:

```python
@staticmethod
def _normalized_scores(candidates: list[Candidate]) -> dict[str, float]:
    values = {
        item.parent_asin: max(float(item.score), 0.0) for item in candidates
    }
    maximum = max(values.values(), default=0.0)
    if maximum <= 0:
        return {identifier: 0.0 for identifier in values}
    return {identifier: value / maximum for identifier, value in values.items()}
```

- [ ] **Step 4: Verify GREEN and the exact-before-relaxed invariant**

Run:

```powershell
& $python -m pytest `
  tests/unit/test_ranker.py `
  tests/unit/test_retrieval.py `
  tests/integration/test_scenarios.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Run S3 development CV and decide keep/reject**

Run the shared folds 1-4 command. Keep fusion `0.10` only if selection score improves by at least `0.002` over S2, mean does not fall, no fold falls by more than `0.02` versus S0, and scenario safeguards remain satisfied.

If rejected, restore `rank_fusion_weight=0.0`, remove only the score-order expectation that assumes active fusion while retaining constructor/range coverage, rerun the focused tests, and record `fusion_weight=0.10 rejected` in the final analysis.

- [ ] **Step 6: Commit only the accepted configuration**

```powershell
git add `
  src/compasscart/config.py `
  src/compasscart/agent.py `
  src/compasscart/ranker.py `
  tests/unit/test_ranker.py
git commit -m "feat: restore bounded fusion evidence in ranking"
```

If fusion is rejected and production behavior remains at `0.0`, use commit message `test: bound optional rank fusion`.

### Task 6: Run A Bounded Browsing-MMR Experiment

**Files:**
- Modify: `src/compasscart/config.py`
- Modify: `tests/unit/test_models.py`

- [ ] **Step 1: Write the failing candidate-value test**

Add to `tests/unit/test_models.py`:

```python
def test_runtime_config_uses_relevance_first_browsing_candidate():
    assert RuntimeConfig().mmr_lambda == 1.0
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
& $python -m pytest tests/unit/test_models.py -q
```

Expected: failure because the S3 configuration still uses `0.85`.

- [ ] **Step 3: Set the single bounded MMR candidate**

Change only:

```python
mmr_lambda: float = 1.0
```

Do not change the MMR algorithm or add further values.

- [ ] **Step 4: Verify local behavior**

Run:

```powershell
& $python -m pytest `
  tests/unit/test_models.py `
  tests/unit/test_ranker.py `
  tests/integration/test_scenarios.py -q
```

Expected: all selected tests pass. If the existing diversity test depends on the direct ranker default, keep the direct default at `0.85`; only Agent runtime configuration changes.

- [ ] **Step 5: Run S4 development CV and decide keep/reject**

Run the shared folds 1-4 command. Keep `1.0` only if selection score improves by at least `0.003` over S3, Browsing HitRate does not fall by more than `0.025`, overall MRR does not fall by more than `0.005`, and all fold safeguards remain satisfied.

If rejected, restore `mmr_lambda=0.85`, remove the candidate-value test, rerun the focused suite, and record the rejected result. This experiment must not delay final validation.

- [ ] **Step 6: Commit only if accepted**

```powershell
git add src/compasscart/config.py tests/unit/test_models.py
git commit -m "perf: prefer relevance in browsing rank order"
```

If rejected, make no S4 commit.

### Task 7: Run Full Validation, Rescore, And Analyze Remaining Gaps

**Files:**
- Create at runtime: `var/score-optimization/final-results.json` (ignored artifact)
- Create at runtime: `var/score-optimization/result-diff.json` (ignored artifact)
- Create at runtime: `var/score-optimization/cv/*.json` (ignored artifacts)
- Create: `reports/final/score-optimization-results.md`

- [ ] **Step 1: Run the complete static and test gate**

Run:

```powershell
& $ruff check src tests tools agent.py
& $python -m pytest -q
```

Expected: Ruff exits `0`; pytest reports zero failures across the full suite.

- [ ] **Step 2: Re-run final development CV**

Run the shared folds 1-4 command once on the final candidate. Read the newly written report and verify all final design gates against S0:

```powershell
$cv = Get-ChildItem var\score-optimization\cv\cv-*.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$report = Get-Content -Raw $cv.FullName | ConvertFrom-Json
$report | Select-Object mean_technical_score,std_technical_score,selection_score |
  Format-List
$report.folds | ForEach-Object {
  [pscustomobject]@{
    fold = $_.fold
    score = $_.aggregate.recommended_technical_score
    hit_rate = $_.aggregate.hit_rate_at_10
    mrr = $_.aggregate.mrr
    mttc = $_.aggregate.mttc
    p95_ms = $_.latency_ms.p95
  }
} | Format-Table -AutoSize
```

Expected: selection score at least `0.510756`, mean at least `0.529195`, and no prohibited fold/scenario regression.

- [ ] **Step 3: Run the unchanged official 200-session evaluator**

Run:

```powershell
New-Item -ItemType Directory -Force var\score-optimization | Out-Null
& $python -m tools.run_agent `
  --catalog data/catalog.jsonl `
  --dataset data/public_set.jsonl `
  --output var/score-optimization/final-results.json
```

Expected: evaluator exits `0` and prints aggregate plus all four scenario metrics.

- [ ] **Step 4: Produce the same-host baseline diff**

Use the fresh baseline created before implementation if it still exists:

```powershell
$baseline = Join-Path $env:TEMP "compasscart-baseline-20260826.json"
if (-not (Test-Path -LiteralPath $baseline)) {
    throw "Fresh Windows baseline is missing; recreate it from tag compasscart-v2 in a clean temporary worktree before comparing."
}
& $python -m tools.compare_results `
  $baseline `
  var/score-optimization/final-results.json `
  --output var/score-optimization/result-diff.json
```

Expected: the diff reports metric deltas, recovered/regressed IDs, rank/turn changes, and scenario counts. Do not compare only against the macOS attachment because platform-sensitive dense boundaries would confound the result.

- [ ] **Step 5: Verify offline fallback, performance, and package contracts**

Run:

```powershell
& $python -m pytest `
  tests/integration/test_fallbacks.py `
  tests/performance/test_runtime.py `
  tests/contract/test_agent_contract.py `
  tests/contract/test_submission_package.py -q
& $python -m tools.package_submission
```

Expected: zero failures, generated package passes its allowlist/checksum smoke test, and no public labels/evaluator files enter the ZIP.

- [ ] **Step 6: Measure cache memory and latency safeguards**

Use the CV report's P95 values and the existing full-catalog smoke measurement method documented in `README.md`. Reject or compact the searchable cache if single-turn P95 exceeds `1.5 s` or cold-start memory rises more than 15% over the documented `576.3 MiB` peak (`662.745 MiB`).

- [ ] **Step 7: Write the evidence report with actual outputs**

Create `reports/final/score-optimization-results.md` only after Steps 1-6. It must contain these values copied exactly from the generated JSON rather than estimates:

```markdown
# CompassCart Score Optimization Results

## Selection Evidence

- Baseline folds 1-4 mean/std/selection
- Final folds 1-4 mean/std/selection
- Per-fold TechnicalScore and P95 latency

## Official Public Evaluation

- Same-host baseline and final TechnicalScore, HitRate@10, MRR, MTTC
- Absolute metric deltas
- Buying, Browsing, Intent Override, and Boundary before/after metrics

## Session Changes

- Recovered and regressed counts by scenario
- Rank/turn improved and worsened counts

## Accepted And Rejected Experiments

- S1 parser/route result
- S2 shared semantics result
- S3 fusion result and final setting
- S4 MMR result and final setting

## Remaining Opportunities

- Remaining misses blocked by constraints
- Remaining misses present below rank 10
- Remaining misses absent from all candidate sources
- Cross-platform deterministic-order risk

## Verification

- pytest and Ruff results
- fallback/package results
- latency and memory results
```

Every bullet must be replaced by measured data and concrete counts. Do not include the diagnostic `0.67` estimate as an achieved result.

- [ ] **Step 8: Commit the final evidence report**

```powershell
git add reports/final/score-optimization-results.md
git commit -m "docs: record score optimization evidence"
```

- [ ] **Step 9: Review the final diff and repository status**

Run:

```powershell
git status --short
$base = git merge-base origin/main HEAD
git diff --check "$base..HEAD"
git log -8 --oneline --decorate
```

Expected: only intended code, tests, plan/spec, and final evidence are committed. The pre-existing untracked `docs/compasscart-agent-architecture-analysis.docx` remains untouched and uncommitted.

## Completion Criteria

- [ ] Parser category spans no longer create unsupported nested style/brand hard constraints.
- [ ] No-preference preserves the current route.
- [ ] Unknown clarification text is soft evidence and remains in bounded query history.
- [ ] Category filtering, attribute candidates, fallback, and ranker use one token-union meaning.
- [ ] Open clarification matching uses cached searchable terms and never changes message hard constraints.
- [ ] S3/S4 experiments are kept only when their independent CV gates pass.
- [ ] Final folds 1-4 selection score is at least `0.510756` and mean is at least `0.529195`.
- [ ] Full pytest, Ruff, fallback, performance, contract, and package gates pass.
- [ ] Same-host official evaluator results and session-level changes are recorded.
- [ ] Remaining optimization opportunities are evidence-backed rather than inferred from aggregate score alone.
