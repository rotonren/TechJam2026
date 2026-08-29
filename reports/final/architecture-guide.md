# CompassCart: How It Works

A working guide for the team. `architecture.md` is the one-page judging map;
this is the longer version you need to change something, run an experiment, or
answer a question about the system in the final.

## The shape of one turn

The evaluator calls `Agent.respond(session_id, message, turn, 10)` and gets back
a message, an `ask_attribute`, and ten `parent_asin` values. Inside, seven
stages run in order. Every one of them is wrapped so a failure degrades rather
than breaks the turn.

```
message
   │
   ▼
1  SessionStore ─ parse the turn into the constraint ledger
   │              versioned, bounded, override-aware
   ▼
2  PolicyMemory ─ observe whether the previous question was answered
   │              (this is where cross-session learning happens)
   ▼
3  RoutePlanner ─ choose Buying or Browsing, and the fusion weights
   │
   ▼
4  HybridRetriever ─ lexical + attribute + profile + dense, fused by weighted RRF
   │                 ~500 candidates
   ▼
5  ConstraintRanker ─ hard-constraint coverage, quality, browsing diversity
   │
   ▼
6  RerankStage ─ rescore the head of the list, per route
   │             Browsing: phrase adjacency.  Buying: off by default.
   ▼
7  QuestionPolicy + StrategySelector ─ what to ask, or whether to ask at all
   │
   ▼
   ResponseBuilder ─ ten unique catalog-valid IDs, token usage, message
```

## What each stage is for

| Stage | File | Its one job |
| --- | --- | --- |
| `SessionStore` | `state.py` | Keep a versioned ledger of what the shopper wants. An override supersedes rather than appends. |
| `MessageParser` | `parser.py` | Turn a message into typed constraints: attribute, value, hardness, operator, source. |
| `PolicyMemory` | `evolution.py` | Estimate how often each attribute is answerable, from observation rather than from our guess. |
| `RoutePlanner` | `router.py` | Decide Buying or Browsing from constraint specificity, and pick fusion weights. |
| `HybridRetriever` | `retrieval.py` | Produce candidates from four independent sources and fuse them by weighted reciprocal-rank. |
| `ConstraintRanker` | `ranker.py` | Score candidates against the ledger; keep exact matches ahead of disclosed relaxations. |
| `RerankStage` | `rerank.py` | Reorder the head of the list on a signal the term-level stages cannot see. |
| `StrategySelector` | `orchestration.py` | Decide what to do with a turn the question policy declined. |
| `QuestionPolicy` | `question_policy.py` | Pick the clarification with the highest expected conversion gain. |
| `ResponseBuilder` | `response.py` | Emit a contract-valid response: ten unique valid IDs, honest token counts. |

## The three stages added in this round

### RerankStage — the phrase-adjacency reranker

Fusion and constraint scoring both work at term level. A requirement stated as
a contiguous phrase - "water resistant rubber outsole" - scores no better
against a product whose title contains that exact phrase than against one that
mentions the same words scattered across its description. This stage supplies
that missing signal.

It reorders only the head of the list, never changes membership, and keeps
exact candidates ahead of disclosed relaxations.

**It runs on Browsing only.** The same stage measured `+0.038` Browsing
HitRate and `-0.025` Buying. A Buying turn already carries explicit hard
constraints, so the constraint ranker is well informed and reordering it
destroys good work.

### PolicyMemory — question priors that correct themselves

`QuestionPolicy` weights each candidate question by how often a shopper can
answer it. That table was hand-written and nothing checked it. Now the agent
treats it as a prior and updates a posterior from whether its own questions
were answered.

The first session behaves exactly as the hand-written table did. Estimates
refine in order of evidence: pooled, then per route, then per shopper segment,
each level applying only once its own bucket clears an observation floor.

It found the two priors we had backwards. `feature` was ranked second-lowest at
`0.70` and is the most productive question at `0.973`; `budget` was top tier at
`0.90` and produced nothing in 16 attempts. That turned out to be a fact about
the catalog: a `parent_asin` is a parent product, not a size or colour SKU, so
size is absent from 91.3% of products and cannot be answered however it is
phrased.

Learning never touches ground truth. The agent observes whether its own
question was productive, never whether its recommendations were right.

### StrategySelector — what to do with a wasted turn

When the question policy has nothing worth asking, the turn is spent: the
shopper replies "those options are not quite right yet, ask me about one
specific attribute" and discloses nothing. `open_probe` asks an open question
instead; `exploit` stops asking once the pool is small enough that a question
cannot pay for its turn.

`open_probe` fires eleven times in 536 turns and is worth most of `+0.018`.
Firing count is a poor proxy for value here.

## Configuration

Every default below is what the organizer will run. Each layer has an ablation
switch, and turning all three off reproduces the pre-round score `0.761209`
exactly.

| Setting | Default | What it does |
| --- | --- | --- |
| `rerank_enabled` | `True` | Master switch for stage 6 |
| `rerank_backend` | `"phrase"` | Browsing-route backend: `phrase`, `cross_encoder`, `llm` |
| `rerank_window` | `50` | Candidates reordered on the Browsing route |
| `rerank_weight` | `0.8` | How far the rerank score moves the ranker's order |
| `rerank_buying_backend` | `None` | Buying-route backend; `None` reuses the Browsing one |
| `rerank_buying_window` | `None` | Buying-route window; `None` reuses `rerank_window` |
| `rerank_buying_weight` | `0.0` | **Zero: the stage does not run on Buying** |
| `rerank_buying_requires_override` | `False` | Gate the Buying backend to override turns only |
| `rerank_prompt_style` | `"flat"` | Model prompt form: `flat`, `structured`, `adaptive` |
| `rerank_max_length` | `128` | Cross-encoder sequence length |
| `evolution_enabled` | `True` | Master switch for the policy memory |
| `strategy_enabled` | `True` | Master switch for the strategy selector |

Environment switches, for ablation without editing code:

| Variable | Effect |
| --- | --- |
| `COMPASSCART_DISABLE_RERANK=1` | Stage 6 off |
| `COMPASSCART_DISABLE_EVOLUTION=1` | Policy memory returns the hand-written prior |
| `COMPASSCART_DISABLE_DENSE=1` | Lexical retrieval only; **scores identically and uses 156 MiB less** |
| `COMPASSCART_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | Credentials for the optional model backend |

## What degrades to what

Nothing in the optional path can break a turn.

| Failure | Result |
| --- | --- |
| No LLM credentials | Backend construction returns the phrase backend |
| LLM timeout, refusal, unparseable reply, or a reply that is not a permutation | That turn is not reordered; after three failures the backend is disabled |
| Cross-encoder asset missing or corrupt | Phrase backend |
| Dense asset missing, corrupt, or failing inference | Pure-Python lexical retrieval |
| FTS5 unavailable | Pure-Python catalog scan |
| Parser, router, retriever, ranker, reranker, question, strategy, or trace raises | That component is skipped and named in the trace's `fallbacks` |
| Rerank window has no distinguishable candidate | The ranker's order is kept, not shuffled onto the identifier tie-break |

**One known gap.** Construction-time failure falls back to the phrase backend;
a runtime failure only disables reranking on that route. With the shipped
defaults this is harmless - the LLM is only ever configured on Buying, where
the default does not rerank at all - but the two paths do not degrade to the
same place, and that is luck rather than design.

## Resource profile

Measured per process, each configuration in its own run.

| Configuration | Peak RSS | Agent alone | Init | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Default | 749.7 MiB | 409.6 MiB | 24418 ms | 0.822490 |
| `COMPASSCART_DISABLE_DENSE=1` | 593.2 MiB | 353.7 MiB | 19653 ms | 0.822490 |

Roughly 240 MiB of the peak is the harness's own copy of the catalog, which any
submission pays for. Dense retrieval is gated to semantic rescue and never
fired across the 536 turns of the public evaluation, which is why disabling it
scores identically - it is kept because the private split may exercise the
rescue path.

The agent requires no network access and reports zero tokens on the default
path.

## Running things

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m tools.run_agent `
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl `
  --output results.json --evidence-output results-evidence.json
```

| Task | Command |
| --- | --- |
| Full test suite | `python -m pytest -q` |
| Lint | `python -m ruff check src tests tools` |
| Frozen-input checks | `python -m tools.verify_frozen_inputs` |
| Cross-validation folds | `python -m tools.run_cv --folds 1 2 3 4 5 --seed 2026` |
| Resource benchmark | `python -m tools.benchmark_release --trials 3` |

## Where to read next

- `reports/final/rerank-results.md` — every rerank experiment, including the
  four that were rejected and why
- `reports/final/evolution-results.md` — the policy memory and strategy
  selector, their ablations, and what the learned priors revealed
- `docs/attribute_schema.md` — the layered attribute schema and why catalog
  discovery is opt-in
- `reports/final/approach-evolution.md` — how the design arrived here, in order
- `reports/final/validation-evidence.md` — every claim above with the command
  that reproduces it: frozen-input checks, cross-validation, the sealed fold,
  the learning-layer control, and the ablation ladder
