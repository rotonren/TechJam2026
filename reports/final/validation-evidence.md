# Validation Evidence

Every number here comes from the unchanged official evaluator. This file exists
so a reviewer can check the claims in `approach-evolution.md` without rerunning
anything, and reproduce any of them with the command printed beside it.

## 1. The inputs are the organizer's, unmodified

```
python -m tools.verify_frozen_inputs
```

| File | Status |
| --- | --- |
| `data/catalog.jsonl` | ok |
| `data/public_set.jsonl` | ok |
| `evaluator/local_evaluator.py` | ok |
| `assets/SHA256SUMS` | ok |
| `assets/model/model.int8.onnx` | ok |
| `assets/model/tokenizer.json` | ok |
| `assets/product_vectors/product_ids.npy` | ok |
| `assets/product_vectors/scales.npy` | ok |
| `assets/product_vectors/vectors.int8.npy` | ok |

9/9. No score in this repository was produced against an edited evaluator, an
edited dataset, or a regenerated asset.

## 2. The headline score

```
python -m tools.run_agent --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output results.json
```

**0.822490** over all 200 sessions and 536 turns, offline, zero tokens, zero
fallbacks.

That number is the whole public set, which is also the set every design decision
was made against. On its own it is not evidence of generalization. Sections 3
and 4 are.

## 3. Cross-validation, and a fold that was never looked at

```
python -m tools.run_cv --folds 1 2 3 4 --seed 2026
python -m tools.run_cv --folds 5 --seed 2026 --audit
```

Folds 1-4 were the tuning folds: every experiment in `rerank-results.md` and
`evolution-results.md` was read on them. **Fold 5 was sealed at the start of the
round and run once, at the end, after the configuration was frozen.**

| Fold | n | TechnicalScore | Hit@10 | MRR | MTTC | Fallbacks | p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40 | 0.817467 | 0.9750 | 0.538224 | 2.575 | 0 | 190 ms |
| 2 | 40 | 0.824997 | 0.9750 | 0.569990 | 2.675 | 0 | 323 ms |
| 3 | 40 | 0.769402 | 0.9250 | 0.503006 | 3.200 | 0 | 207 ms |
| 4 | 40 | 0.836976 | 1.0000 | 0.581587 | 2.875 | 0 | 200 ms |
| **mean 1-4** | | **0.812211** | | | | 0 | |
| std | | 0.025676 | | | | | |
| **5 (sealed)** | 40 | **0.834997** | 0.9750 | 0.618323 | 2.900 | 0 | 231 ms |

**The sealed fold scored above the tuning folds' mean and above the all-200
score.** If the round had overfit the public set, this is where it would show,
and it does not.

The honest caveat: n=40 per fold, and the tuning folds span 0.769 to 0.837 -
a spread of 0.068, far wider than most of the individual changes in this round.
One fold cannot certify a `+0.004` decision. It can certify that the `+0.061`
of accepted changes did not come from memorizing 200 sessions, and that is what
it is being used for.

Fold 3 is the hard fold under every configuration measured, tuned or not.

## 4. The learning layer, isolated under cross-validation

The concern with cross-session memory is that it learns the public set rather
than the catalog. The control disables it and changes nothing else:

```
COMPASSCART_DISABLE_EVOLUTION=1 python -m tools.run_cv --folds 1 2 3 4 --seed 2026
```

| Fold | Learning on | Learning off | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 0.817467 | 0.801905 | **+0.015562** |
| 2 | 0.824997 | 0.818863 | **+0.006134** |
| 3 | 0.769402 | 0.771280 | −0.001878 |
| 4 | 0.836976 | 0.778923 | **+0.058053** |
| **mean** | **0.812211** | **0.792743** | **+0.019468** |
| std | 0.025676 | 0.018827 | |
| selection | 0.799373 | 0.783329 | +0.016044 |

Each fold builds a fresh agent and sees only 40 sessions, so the memory operates
on roughly a fifth of the observations it gets in the full run. **It is still
worth `+0.019` there**, which answers the question the CV was run to answer: the
gain is not an artifact of accumulating 200 sessions of the public set.

Where the gain comes from, averaged over the four folds:

| | Learning on | Learning off | Delta |
| --- | ---: | ---: | ---: |
| MTTC | 2.831 | 3.300 | **−0.469 turns** |
| MRR | 0.548 | 0.535 | +0.013 |
| Hit@10 | 0.969 | 0.956 | +0.013 |

The layer mostly buys turns, not recall - which is what it was built to do. It
corrects which question gets asked, so the shopper's requirement arrives sooner.

Fold 3 is the exception worth naming. There the memory bought 0.575 turns of
MTTC (worth `+0.0115`) and gave back 0.045 of MRR (worth `-0.0134`), netting
`-0.0019`. Asking a more answerable question got the requirement out sooner but
landed on a slightly worse ordering. **The layer is a net win in 3 folds of 4,
not 4 of 4**, and the report should not round that off.

## 5. Every layer has an off switch, and off reproduces the old score

Each row is one full run of the 200-session set, differing from the row above it
by one setting. `COMPASSCART_DISABLE_RERANK`, `COMPASSCART_DISABLE_EVOLUTION`,
and `strategy_enabled` toggle the three layers without editing code; the
intermediate rows set `rerank_buying_weight` and the memory's conditioning level
in `AgentConfig`.

| Configuration | TechnicalScore |
| --- | ---: |
| All three layers off | 0.761209 |
| + phrase rerank, one weight everywhere | 0.771831 |
| + rerank on Browsing only | 0.783514 |
| + policy memory, pooled | 0.800849 |
| + policy memory, per route | 0.804724 |
| + strategy selection — **shipped default** | **0.822490** |
| + LLM rerank on Buying (needs credentials) | 0.826831 |

All-off reproduces `0.761209`, the pre-round score, exactly. That is the check
that the three layers are the only thing that changed.

## 6. What was measured and rejected

Kept in the tree, disabled, and tested, because they are the evidence behind the
claims above:

| Change | Delta | Why it lost |
| --- | ---: | --- |
| Window-local IDF weighting | −0.011 | Rescoring a window by statistics of that window |
| ONNX cross-encoder backend | −0.007 | Carries the window confound of §7; not re-measured |
| `relax` strategy | −0.010 | A stall does not mean the filters are wrong |
| Constraint-count disclosure signal | −0.015 | Counts the parser, not the shopper |
| Grouped model prompt | −0.003 | Inverted a label; see `approach-evolution.md` §7 |
| State-selected prompt | +0.0002 | Below noise; the per-route optima did not compose |
| Layered catalog discovery | ±0.000 | Identical vocabulary for +92.7 s of startup |

## 7. One measurement that was wrong, and its correction

The first LLM result reported `−0.001266`. The model ran at rerank window 20
while the baseline ran at 50, and the window was stage-level rather than
per-route, so it moved for the phrase backend too.

```
window 50, no model      0.822490
window 20, no model      0.812999    <- the control that was missing
```

The window alone cost `−0.0095`, and all of it had been charged to the model.
With the window made per-route, the model measures **`+0.004341`**.

The corrected number is the one reported. The uncorrected one is recorded here
because the failure mode - one run, two variables - is the reason every other
row in this file has a named control beside it.

## 8. Cost and runtime

| | Value |
| --- | ---: |
| API cost, all CV runs | $0.00 |
| API cost, shipped default | $0.00 |
| Network calls, shipped default | 0 |
| Reported tokens, shipped default | 0 |
| p95 turn latency | 190-323 ms |
| Init | 19.6 s |
| Peak RSS | 749.7 MiB (593.2 MiB with dense disabled) |
| Tests | 1047 passing |
| Lint | `ruff check src tests tools` clean |

The shipped configuration needs no network access and no credentials. The LLM
path is opt-in via environment variables, per `docs/submission_rules.md`, which
forbids shipping API keys and warns that official scoring may run without
network access.
