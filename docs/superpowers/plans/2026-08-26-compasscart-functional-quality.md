# CompassCart Functional Quality Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the existing CompassCart Agent so explicit shopping constraints are interpreted correctly, preserved across turns, enforced consistently by every retrieval source, and handled predictably when the user changes intent.

**Architecture:** Keep the current offline Agent contract and component boundaries, but introduce one shared constraint-semantics layer used by parsing, state, retrieval, ranking, and fallback selection. Preserve the deterministic lexical path as the baseline, add catalog-derived aliases around it, and make every relaxation of an impossible request explicit in trace/message output. Product UI, payment, persistence, and repository delivery are intentionally outside this plan.

**Tech Stack:** Python 3.10+, standard library (`dataclasses`, `json`, `re`, `sqlite3`), optional NumPy/ONNX Runtime/tokenizers, pytest, and the existing deterministic evaluator.

---

## Scope and behavioral contract

This plan changes the behavior of the existing competition Agent only. It does not change the public method signatures:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict: ...
```

The response must continue to contain exactly the four keys already asserted by the contract tests: `message`, `ask_attribute`, `recommendations`, and `usage`. Explanations and relaxation evidence belong in the natural-language message, trace records, and the CLI formatter; they must not add unapproved response keys.

The first implementation pass has four required outcomes:

1. An explicit hard constraint is never silently violated while an exact catalog match exists.
2. Budget operators, alternatives, negative preferences, and common catalog values have unambiguous semantics.
3. A new turn is not lost merely because the rule parser does not recognize every word.
4. An explicit intent change starts a predictable new constraint context and does not inherit stale questions or filters accidentally.

The following are deliberately deferred: a browser UI, HTTP service, durable sessions, payment/order actions, image search, and multilingual support beyond the English competition language. They should be planned separately after this pass is evaluated.

## File map

| File | Responsibility after this plan |
| --- | --- |
| `src/compasscart/constraints.py` | Shared operators, normalization, product matching, conflict/violation reporting |
| `src/compasscart/models.py` | Constraint and candidate fields required by the shared semantics |
| `src/compasscart/parser.py` | English intent, slot, budget, negation, alternative, and override parsing |
| `src/compasscart/normalization.py` | Alias/token normalization and catalog-derived canonical values |
| `src/compasscart/state.py` | Per-intent ledger transitions, raw query evidence, profile soft preferences |
| `src/compasscart/agent.py` | Orchestrates the revised parser, plan, response, and trace fields |
| `src/compasscart/router.py` | Explicit browsing/buying precedence and route explanation |
| `src/compasscart/catalog.py` | Catalog vocabulary and shared hard-filter calls |
| `src/compasscart/retrieval.py` | Source filtering, controlled relaxation, and violation metadata |
| `src/compasscart/ranker.py` | Constraint-aware ranking using the shared matcher |
| `src/compasscart/question_policy.py` | Questions that are answerable and represented by the catalog |
| `src/compasscart/response.py` | Contract-safe messages for exact and relaxed results |
| `src/compasscart/tracing.py` | Route, intent, relaxation, and violation evidence |
| `tools/demo_chat.py` | Human-readable display of exact/relaxed results and constraints |
| `tests/unit/test_constraints.py` | Operator and matcher truth table |
| `tests/unit/test_parser.py` | Parser regression cases |
| `tests/unit/test_state.py` | Ledger and override regression cases |
| `tests/unit/test_router.py` | Route precedence cases |
| `tests/unit/test_retrieval.py` | Hard-filter and fallback guarantees |
| `tests/unit/test_ranker.py` | Ranking and violation ordering |
| `tests/unit/test_question_policy.py` | Supported-question policy |
| `tests/integration/test_scenarios.py` | End-to-end multi-turn behavior |
| `tests/integration/test_fallbacks.py` | Failure and controlled-relaxation behavior |
| `tests/contract/test_agent_contract.py` | Unchanged official response contract |

---

### Task 1: Define one constraint-semantics layer

**Files:**
- Create: `src/compasscart/constraints.py`
- Modify: `src/compasscart/models.py`
- Create: `tests/unit/test_constraints.py`
- Modify: `tests/unit/test_models.py`

- [ ] **Step 1: Add the operator types and compatibility fields**

Keep existing positional construction valid by appending fields with defaults to `Constraint` and `Candidate` rather than reordering existing fields.

```python
# src/compasscart/models.py
from typing import Literal

ConstraintOperator = Literal["eq", "in", "not_in", "lte", "gte", "between"]

@dataclass(frozen=True)
class Constraint:
    attribute: str
    value: str
    confidence: float
    is_hard: bool
    source: ConstraintSource
    created_turn: int
    intent_version: int
    status: ConstraintStatus = "active"
    operator: ConstraintOperator = "eq"
    upper_value: str | None = None
    alternatives: tuple[str, ...] = ()

    def values(self) -> tuple[str, ...]:
        return self.alternatives or (self.value,)

@dataclass
class Candidate:
    parent_asin: str
    product: dict[str, object] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    violations: tuple[str, ...] = ()
    relaxed: bool = False
```

`operator="in"` represents an explicit OR such as “black or blue”; `between` uses `value` as the lower bound and `upper_value` as the upper bound. Existing equality constraints remain unchanged at call sites until the parser migration task.

- [ ] **Step 2: Implement the shared matcher**

Create these exact functions in `src/compasscart/constraints.py`:

```python
def matches_constraint(
    product: dict[str, object],
    attributes: dict[str, tuple[str, ...]],
    constraint: Constraint,
) -> bool: ...

def hard_constraint_violations(
    product: dict[str, object],
    attributes: dict[str, tuple[str, ...]],
    constraints: list[Constraint],
) -> tuple[str, ...]: ...

def display_constraint(constraint: Constraint) -> str: ...
```

The matcher must apply these rules:

- `eq` matches one normalized catalog value.
- `in` matches when any alternative is present.
- `not_in` matches only when none of the alternatives is present.
- `lte` and `gte` compare numeric `price` values; missing or nonnumeric prices do not match.
- `between` is inclusive at both endpoints.
- A missing catalog attribute is a mismatch for a hard constraint, not an implicit match.

`hard_constraint_violations` returns stable strings such as `budget<=50.00` and `color in (black,blue)` in input order. It is the only function retrieval and ranking code may use to decide whether a hard condition matches.

- [ ] **Step 3: Write the operator truth-table tests**

Add a small in-memory product matrix covering:

```python
PRODUCT = {
    "price": 75.0,
    "details": {"Color": "Blue", "Material": "Cotton"},
}
ATTRIBUTES = {"color": ("blue",), "material": ("cotton",)}
```

Assert that `lte(80)` and `between(50, 80)` match, `gte(80)` does not, `in(black, blue)` matches, `not_in(red)` matches, and a missing attribute fails. Assert that `display_constraint` is deterministic.

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_constraints.py tests/unit/test_models.py -q
```

Expected: all new tests pass and all pre-existing model tests remain green.

---

### Task 2: Expand parsing without losing free-text evidence

**Files:**
- Modify: `src/compasscart/parser.py`
- Modify: `src/compasscart/normalization.py`
- Modify: `src/compasscart/catalog.py`
- Modify: `src/compasscart/agent.py`
- Modify: `src/compasscart/state.py`
- Modify: `tests/unit/test_parser.py`
- Modify: `tests/unit/test_state.py`

- [ ] **Step 1: Extend parser/state records with operator and continuation fields**

Append `operator`, `upper_value`, and `alternatives` to `ParsedConstraint` with the same defaults as `Constraint`. Add `is_continuation: bool = False` to `ParseResult`, set it for `show me more`, `more options`, and `different choices`, and add `continuation_requested: bool = False` to `SessionState`. Update `SessionStore._to_constraint` to copy the operator fields exactly. Do not encode operators into the display text or into raw query strings.

- [ ] **Step 2: Parse explicit budget operators and alternatives**

Replace the single amount-only budget extraction with phrase-specific patterns and these mappings:

| User phrase | Parsed operator |
| --- | --- |
| `under`, `below`, `less than`, `at most`, `up to` | `lte` |
| `over`, `above`, `more than`, `at least`, `from` | `gte` |
| `between A and B`, `from A to B` | `between(A, B)` |
| `not`, `without`, `no` + known value | `not_in` |
| `black or blue` | one `in(black, blue)` constraint |

Do not emit two same-attribute hard constraints for one OR expression. Preserve the existing `$80`/`under $80` behavior as `lte(80)`.

- [ ] **Step 3: Add catalog-derived aliases**

Give `MessageParser` an optional immutable vocabulary built by `CatalogIndex` from normalized `attribute_inverted` keys. Use longest-first phrase matching for `brand`, `size`, `category`, `material`, `style`, `feature`, and `use_case`, while retaining the existing fixed vocabularies as a fast path. Add singular/plural normalization for categories and common punctuation such as `women's`.

The constructor shape becomes:

```python
MessageParser(vocabulary: Mapping[str, tuple[str, ...]] | None = None)
```

`CompassCartAgent` must construct the catalog before the parser and pass `self.catalog.parser_vocabulary()` to it. The vocabulary is read-only after construction and must not perform network calls.

- [ ] **Step 4: Parse all recognizable attributes before resolving a pending question**

When `expected_attribute` is present, first extract any known color/material/size/brand/category/budget/etc. from the complete message. Use `expected_attribute` only for the remaining unrecognized text. For example, a reply to a material question saying `blue shoes` must produce a color constraint and a category constraint; it must not store the entire phrase as material.

If the remaining text is empty or clearly unrelated, return no forced hard constraint and keep the raw text for lexical/dense retrieval.

- [ ] **Step 5: Preserve every non-empty user turn as bounded query evidence**

Change `SessionStore.update` to append each non-empty message to the last four `query_history` entries, regardless of whether the parser found a constraint. On an explicit override, clear old query evidence first and append only the override message. Change `CompassCartAgent._query_text` to include the current message exactly once, normalized and bounded, instead of discarding it with `del message`.

- [ ] **Step 6: Add parser and query-preservation regressions**

Add tests with these exact expectations:

```python
parser = MessageParser(vocabulary={
    "brand": ("nike",),
    "size": ("10",),
    "category": ("shoes",),
})

over = parser.parse("I need shoes over $50", 1)
assert [(item.attribute, item.operator, item.value) for item in over.constraints] == [
    ("budget", "gte", "50.00"),
    ("category", "eq", "shoes"),
]

between = parser.parse("I need shoes between $50 and $100", 1)
budget = next(item for item in between.constraints if item.attribute == "budget")
assert (budget.operator, budget.value, budget.upper_value) == (
    "between", "50.00", "100.00"
)

alternatives = parser.parse("I need black or blue shoes", 1)
color = next(item for item in alternatives.constraints if item.attribute == "color")
assert color.operator == "in"
assert color.alternatives == ("black", "blue")

negative = parser.parse("I do not want red shoes", 1)
color = next(item for item in negative.constraints if item.attribute == "color")
assert color.operator == "not_in"
assert color.value == "red"

brand = parser.parse("I want Nike shoes", 1)
assert any(item.attribute == "brand" and item.value == "nike" for item in brand.constraints)

size = parser.parse("I need size 10 shoes", 1)
assert any(item.attribute == "size" and item.value == "10" for item in size.constraints)
```

Also test that an unrecognized second-turn phrase appears in `state.query_history` and in the next retrieval query, and that a pending material question does not swallow a recognizable color/category answer.

- [ ] **Step 7: Run focused parser/state tests**

Run:

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_parser.py tests/unit/test_state.py tests/unit/test_catalog.py -q
```

Expected: all focused tests pass, with no change to the four-key Agent response contract.

---

### Task 3: Make intent transitions and profile personalization explicit

**Files:**
- Modify: `src/compasscart/state.py`
- Modify: `src/compasscart/models.py`
- Modify: `src/compasscart/parser.py`
- Modify: `tests/unit/test_state.py`
- Modify: `tests/integration/test_scenarios.py`

- [ ] **Step 1: Distinguish attribute correction from a new shopping goal**

Add `override_scope: Literal["none", "goal", "attribute"] = "none"` to `SessionState` and reset it to `"none"` at the start of every `SessionStore.update` call.

Use the parsed override marker and category evidence to choose one of two transitions:

1. An override that names a new category starts a new goal: supersede all prior user/clarification hard constraints, clear query history, clear pending/asked/no-preference attributes, then apply the new message and retain profile constraints only as soft preferences.
2. An override that names no new category changes only the explicitly mentioned attributes and retains compatible constraints such as the existing category.

Record the chosen transition on `SessionState.override_scope` for the current turn and copy it into the trace as `override_scope="goal"` or `override_scope="attribute"`. Never silently retain a conflicting hard constraint.

- [ ] **Step 2: Version question state**

Either clear `asked_attributes`, `pending_attribute`, and `no_preference_attributes` on a goal-level override or store them keyed by `intent_version`. The observable rule is that a new goal may ask an attribute that was rejected or asked under the previous goal, while the same goal never repeats it.

- [ ] **Step 3: Expand profile soft constraints safely**

Keep all profile-derived constraints `is_hard=False` and map these canonical tags before falling back to a feature value: `comfort`, `comfortable`, `fit`, `durability`, `durable`, `warmth`, `warm`, `weather`, `weatherproof`, `lightweight`, `breathable`, `waterproof`, and `stretch`. Use `purchase_frequency`, `average_prior_rating`, and `rating_style` only as quality/profile priors; they must never become hard filters.

- [ ] **Step 4: Add transition regressions**

Test all of the following:

- `red cotton dress` followed by a goal-level `actually, I need a black leather belt` leaves no active red, cotton, or dress constraint.
- `red cotton dress` followed by an attribute-level `actually, make it blue` retains category dress and material cotton while replacing red.
- A new goal can ask an attribute rejected in the old goal.
- A profile color never overrides an explicit user color.
- Profile numeric/rating fields change only soft ranking, not hard candidate eligibility.

- [ ] **Step 5: Run state and scenario tests**

Run:

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_state.py tests/integration/test_scenarios.py -q
```

Expected: old-intent constraints are absent from the active ledger after a goal-level override and every response remains catalog-valid.

---

### Task 4: Enforce hard filters across routing, retrieval, ranking, and fallback

**Files:**
- Modify: `src/compasscart/router.py`
- Modify: `src/compasscart/catalog.py`
- Modify: `src/compasscart/retrieval.py`
- Modify: `src/compasscart/ranker.py`
- Modify: `src/compasscart/models.py`
- Modify: `src/compasscart/agent.py`
- Modify: `src/compasscart/tracing.py`
- Modify: `tests/unit/test_router.py`
- Modify: `tests/unit/test_retrieval.py`
- Modify: `tests/unit/test_ranker.py`
- Modify: `tests/integration/test_fallbacks.py`

- [ ] **Step 1: Give explicit browsing hints precedence**

Add `route_hint: Route | None = None` to `SessionState`. Set it from `ParseResult.route_hint` on every update, and have `RoutePlanner.build_plan` read `state.route_hint`. Use this precedence:

1. An explicit browsing phrase such as `still exploring` selects Browsing unless the same message contains an explicit hard purchase marker (`must`, `key requirement`, budget ceiling/floor, or `need ... under ...`).
2. An explicit buying marker selects Buying.
3. Query specificity is only a fallback when neither marker exists.

Add a trace field `route_reason` with values `explicit_browsing`, `explicit_buying`, or `specificity_fallback`.

- [ ] **Step 2: Filter every candidate source with the shared matcher**

Apply `hard_constraint_violations` to lexical, attribute, profile, dense, and popularity candidates before RRF. Do not rely on the ranker’s penalty to enforce an explicit hard condition. The same product must receive the same violation result regardless of its source.

- [ ] **Step 3: Implement controlled relaxation with evidence**

When fewer than ten exact candidates exist, keep all exact candidates first. Then append category/popularity alternatives only after strict candidates are exhausted, set `Candidate.relaxed=True`, and record its violation strings. Never replace an exact candidate with a relaxed one merely because its quality score is higher.

Add `relaxed_count` and `relaxed_constraints` to the trace. `ResponseBuilder` must change the message to a clear alternative message when any returned item is relaxed, for example: `I found a few exact matches and added close alternatives after relaxing the budget.`

The Agent must still return at least one valid catalog ID when the catalog is non-empty, preserving the official contract while making the relaxation visible.

- [ ] **Step 4: Make ranker scoring agree with retrieval filtering**

Use `matches_constraint` for hard coverage and conflict calculations. Treat a relaxed candidate as lower priority than every exact candidate before applying quality or diversity. Keep MMR only for Browsing’s final list; do not let diversity displace an exact Buying match.

- [ ] **Step 5: Add hard-filter regression fixtures**

Create a 12-product fixture with four products under `$50`, four between `$50` and `$100`, and four over `$100`. Assert:

- `under $50` returns no over-budget item while exact results exist.
- `between $50 and $100` returns only the inclusive interval before any alternatives.
- `black or blue` returns either color, not only the last parsed color.
- A dense-only candidate that violates budget is excluded before ranking.
- If fewer than ten exact products exist, exact products precede clearly marked alternatives.

- [ ] **Step 6: Run retrieval and fallback tests**

Run:

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_router.py tests/unit/test_retrieval.py tests/unit/test_ranker.py tests/integration/test_fallbacks.py -q
```

Expected: no hard-filter violation occurs while an exact candidate exists, and impossible requests still return valid IDs with a relaxation trace.

---

### Task 5: Make clarification questions answerable and useful

**Files:**
- Modify: `src/compasscart/question_policy.py`
- Modify: `src/compasscart/response.py`
- Modify: `src/compasscart/tracing.py`
- Modify: `tools/demo_chat.py`
- Modify: `tests/unit/test_question_policy.py`
- Modify: `tests/contract/test_agent_contract.py`
- Modify: `tests/unit/test_demo_chat.py`

- [ ] **Step 1: Restrict questions to represented attributes**

Remove `other` from the automatic question candidates unless the catalog has an explicit searchable `other` field and the parser can map the answer to it. Prefer `category`, `budget`, `size`, `brand`, `material`, `color`, `style`, `feature`, and `use_case` only when at least two answerable values appear in the current candidate set.

- [ ] **Step 2: Keep question and answer semantics aligned**

For each question, keep `ask_attribute` as the official machine-readable value and generate a message that names the expected answer. A user answer containing recognizable values for another attribute must be parsed as those values too; do not force it into the pending attribute.

Recognize `show me more`, `more options`, and `different choices` as a continuation request. On that request only, use `SessionState.previous_recommendations` to exclude already shown IDs before filling the list; ordinary refinement turns must continue to allow a previously shown product if it is still the best exact match.

`SessionStore.update` must set `continuation_requested` from the current parse result, and `HybridRetriever.retrieve` must accept the exclusion set explicitly. `CompassCartAgent.respond` clears the flag after building the response so it cannot leak into the next ordinary turn.

- [ ] **Step 3: Explain constrained and relaxed results without changing the contract**

Keep the response keys unchanged. Update the message templates so that:

- exact results use the existing concise recommendation message;
- relaxed results mention which constraint was relaxed;
- an active clarification still appears in the message and `ask_attribute`;
- the CLI trace shows `route`, `intent_version`, `ask_attribute`, `relaxed_count`, and `fallbacks`.

- [ ] **Step 4: Add question regressions**

Assert that the policy never asks `other` for a candidate set with no `other` values, does not repeat a rejected attribute within one intent, can ask it again after a goal-level override, that a `show me more` turn excludes the previous list, and that a normal refinement turn does not exclude it. Keep `set(response) == {"message", "ask_attribute", "recommendations", "usage"}`.

- [ ] **Step 5: Run question and contract tests**

Run:

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_question_policy.py tests/contract/test_agent_contract.py tests/unit/test_demo_chat.py -q
```

Expected: all responses remain contract-valid and every automatic question has a parser/catalog path that can use the answer.

---

### Task 6: Verify behavior against scenarios and the official evaluator

**Files:**
- Modify: `tests/integration/test_scenarios.py`
- Modify: `tests/performance/test_runtime.py`
- Modify: `reports/final/architecture.md` only if the implemented semantics differ from its current description
- Create: `tests/integration/test_functional_edge_cases.py`

- [ ] **Step 1: Add one end-to-end edge-case suite**

The suite must exercise these conversations through `agent.Agent`, not private component methods:

```text
1. “I need black or blue running shoes under $80.”
2. “I’m still exploring shoes.” followed by a material answer.
3. “I need a red cotton dress.” followed by “Actually, I need a black leather belt for work.”
4. “I don’t want leather.”
5. An unknown second-turn phrase followed by a known category.
```

Assert valid unique IDs every turn, correct active constraints, no exact hard-filter violation, expected route reason, and no repeated stale question after an override.

- [ ] **Step 2: Run the complete local quality gate**

Run:

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\ruff.exe check .
```

Expected: pytest exits with code 0. Ruff must either exit with code 0 or have each remaining warning explicitly classified as outside this functional pass before the next task begins.

- [ ] **Step 3: Run development-fold evaluation**

With the frozen catalog and public set available, run:

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m tools.run_cv --catalog data/catalog.jsonl --dataset data/public_set.jsonl --folds 1 2 3 4
```

Compare the new report with the recorded baseline. The release is not accepted if overall TechnicalScore drops materially (more than `0.02`) or if Buying/Boundary validity regresses. The primary target is a measurable improvement in Intent Override HitRate@10 and MTTC, followed by Browsing route accuracy.

- [ ] **Step 4: Run the sealed audit only after the development gate**

Run:

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m tools.run_cv --catalog data/catalog.jsonl --dataset data/public_set.jsonl --folds 5 --audit
```

Record the new route distribution, fallback count, hard-filter violation count, and latency alongside the TechnicalScore. Do not tune against individual sealed samples.

- [ ] **Step 5: Run the manual CLI scenarios**

Run each scenario in lexical mode and inspect that the displayed trace agrees with the Agent response:

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m tools.demo_chat --scenario browsing --lexical
& .\.venv\Scripts\python.exe -m tools.demo_chat --scenario buying --lexical
& .\.venv\Scripts\python.exe -m tools.demo_chat --scenario override --lexical
& .\.venv\Scripts\python.exe -m tools.demo_chat --scenario boundary --lexical
```

Expected: no stale constraint appears after the override, Browsing is labeled Browsing when the user says they are exploring, and any relaxed recommendation is disclosed.

---

## Deferred second-stage product plan

The following are useful if CompassCart is intended to become a real multi-user shopping product, but they are not prerequisites for the competition Agent and should not be mixed into the first implementation pass:

1. Add a response-enrichment adapter that joins ASINs with title, price, rating, matched constraints, and explanation text without changing the official Agent response contract.
2. Add a persistent session repository and a serialized Agent worker for restart recovery and concurrent requests.
3. Add an authenticated HTTP API and browser UI for sessions, feedback, comparison, and “show more”/exclude actions.
4. Add catalog freshness, product links, inventory/price disclaimers, and optional multilingual parsing.

That work should be written as a separate plan after the functional-quality evaluation above, because it introduces different security, persistence, and deployment decisions.

## Completion checklist

- [ ] Shared operators cover equality, alternatives, negation, lower/upper bounds, and inclusive ranges.
- [ ] Every retrieval source and fallback uses the same hard-constraint matcher.
- [ ] Relaxed alternatives are marked and disclosed rather than silently mixed with exact matches.
- [ ] Browsing hints take precedence over generic specificity.
- [ ] Raw multi-turn evidence is retained even when parsing finds no slot.
- [ ] Goal-level overrides clear stale user constraints and per-intent question state.
- [ ] Profile information remains soft and is mapped to searchable catalog attributes.
- [ ] Automatic questions are answerable by the parser and represented in the catalog.
- [ ] Official response keys and valid-ID guarantees remain unchanged.
- [ ] Full tests, lint, development folds, sealed audit, and four CLI scenarios are reviewed before implementation is called complete.
