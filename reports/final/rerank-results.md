# Rerank Stage: Results and Rejected Variants

Recorded on 2026-08-29. Every number here comes from the unchanged official
evaluator (`evaluator/local_evaluator.py`) over the frozen 200-session public
set and the frozen 50,000-product catalog. The parent runtime is commit
`4f56f2f`, whose public TechnicalScore is `0.761209`.

## Why a rerank stage

An oracle probe instrumented `HybridRetriever` and `ConstraintRanker` to record
the target product's position at each pipeline stage across all 200 sessions
(691 turns, zero agent exceptions). Retrieval is not the bottleneck:

| Recall of the target in the retrieval output | @10 | @20 | @50 | @100 | any |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 200 sessions | 0.840 | 0.925 | 0.960 | 0.980 | 0.990 |

Only two sessions never retrieve the target at all. Of the 14 misses at that
time, 12 had the target in the candidate pool between positions 10 and 135. The
target was inside the top-10 of the final list in 186 sessions but at rank 1 in
only 64, so 122 sessions were losing MRR purely to ordering.

The gap the stage targets is specific: fusion and constraint scoring both work
at term level, so a requirement the shopper states as a contiguous phrase
("water resistant rubber outsole") scores no better against a product whose
title contains that exact phrase than against one that mentions the same words
scattered across its description.

## Accepted design

`PhraseMatchBackend` scores each candidate against the phrases the shopper has
actually stated — parsed constraint values *and* raw messages, because much of
what a shopper discloses is free text that no structured attribute captures.
Each phrase contributes token coverage and the length of its longest run
matched contiguously, weighted by evidence type (hard constraint `1.0`, raw
message `0.8`, soft constraint `0.6`, profile `0.2`).

`RerankStage` blends that score with the candidate's existing *rank position*
rather than its raw score, so the constraint ranker's ordering — including its
Browsing diversity pass — survives without any dependence on score scale. The
stage reorders only the head of the list, never changes its membership, keeps
exact candidates ahead of disclosed relaxations, and returns the input
unchanged on an expired deadline, an empty evidence set, or a window in which
no candidate is distinguishable.

### Weight and window

| Window | Weight | TechnicalScore | Hit@10 | MRR | MTTC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.0 | 0.761209 | 0.9300 | 0.489030 | 3.525 |
| 50 | 0.3 | 0.770327 | 0.9200 | 0.534089 | 3.495 |
| 50 | 0.45 | 0.766769 | 0.9200 | 0.521562 | 3.485 |
| 50 | 0.6 | 0.770365 | 0.9300 | 0.515550 | 3.465 |
| **50** | **0.8** | **0.771831** | 0.9350 | 0.517438 | 3.545 |
| 20 | 0.8 | 0.771569 | 0.9400 | 0.499563 | 3.415 |
| 100 | 0.8 | 0.763268 | 0.9250 | 0.512228 | 3.645 |
| 100 | 1.0 | 0.741142 | 0.9100 | 0.476141 | 3.835 |

Weight `0.0` reproduces the parent runtime's `0.761209` exactly, which
confirms the stage is inert when disabled.

### Route separation

Applying one weight to every route hides opposite effects. Against the
`0.761209` parent, the flat `window 50 / weight 0.8` configuration moved the
scenarios in different directions:

| Scenario | Hit@10 | MRR |
| --- | ---: | ---: |
| Browsing | +0.0375 | +0.1006 |
| Buying | −0.0250 | −0.0180 |

A Buying turn already carries explicit hard constraints, so the constraint
ranker is well informed and reordering it mostly destroys good work. A Browsing
turn has vague evidence and gains the most. Giving the Buying route its own
weight recovers that loss:

| Configuration | TechnicalScore | Hit@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Browsing 0.8 / Buying 0.8 | 0.771831 | 0.9350 | 0.517438 | 3.545 |
| Browsing 0.8 / Buying 0.3 | 0.775705 | 0.9350 | 0.521683 | 3.415 |
| **Browsing 0.8 / Buying 0.0** | **0.783514** | **0.9450** | **0.528714** | **3.380** |

Disabling the stage on Buying entirely beats damping it. This is the shipped
default. Per scenario, against the parent runtime:

| Scenario | Hit@10 | MRR | MTTC | Parent MTTC |
| --- | ---: | ---: | ---: | ---: |
| Browsing | 0.9500 | 0.535640 | 3.625 | 3.813 |
| Buying | 0.9375 | 0.467068 | 2.700 | 2.700 |
| Intent Override | 0.9667 | 0.694762 | 4.400 | 4.400 |
| Boundary | 0.9000 | 0.468333 | 3.800 | 5.200 |

Buying and Intent Override are numerically identical to the parent, as
intended — Intent Override sessions route as Buying, so the stage does not run
on them. The gain is Browsing plus a `1.40`-turn MTTC improvement on Boundary.

## Rejected: window-local inverse document frequency

Hypothesis: a contiguous run of terms every candidate shares is weaker
evidence than a run of rare ones, so runs and coverage should be weighted by
local IDF (`log1p(N / df)` over the rerank window).

| Weight | With local IDF | Without | Change |
| ---: | ---: | ---: | ---: |
| 0.3 | 0.758092 | 0.770327 | −0.012235 |
| 0.6 | 0.759382 | 0.770365 | −0.010983 |

Both variants scored below doing nothing at all. The hypothesis is wrong in
this form for a structural reason: the rerank window is selected *by* the
query, so the shopper's own terms are by construction common inside it. A
window-local statistic therefore penalizes exactly the evidence that matters
and promotes incidentally rare terms. Corpus-level IDF would not have this
defect, but BM25 already prices term rarity upstream, so the rerank stage
deliberately contributes adjacency and nothing else. Reverted; the reasoning is
recorded in the `rerank.py` module docstring so it is not retried.

## Rejected: ONNX cross-encoder backend

`cross-encoder/ms-marco-MiniLM-L6-v2`, quantized int8 ONNX (23.2 MB), run
through the existing `onnxruntime` and `tokenizers` dependencies with no new
package and no network at inference time.

| Backend | Window | TechnicalScore | Hit@10 | MRR | MTTC | Eval wall clock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phrase, Buying 0.0 | 50 | **0.783514** | 0.9450 | 0.528714 | 3.380 | 205 s |
| Cross-encoder, Buying 0.3 | 20 | 0.776768 | 0.9300 | 0.537226 | 3.470 | 357 s |
| Cross-encoder, Buying 0.0 | 20 | 0.776227 | 0.9300 | 0.537091 | 3.495 | 288 s |
| Phrase, Buying 0.3 | 50 | 0.775705 | 0.9350 | 0.521683 | 3.415 | 215 s |

The cross-encoder produces the best MRR of any configuration measured
(`0.537226`) and the worst Hit@10 of the four (`0.9300`). Because Hit@10 carries
weight `0.50` against MRR's `0.30`, better ordering does not pay for lost
coverage. The window is not the explanation: the phrase backend scores
`0.771569` at window 20 versus `0.771831` at window 50, a difference of
`0.000262`.

The interpretation is that a model trained on MS MARCO web passages judges
general semantic relevance, whereas this benchmark's relevance signal is
verbatim quotation — the simulated shopper states requirements drawn from the
target product's own `features` and `details` text. Phrase matching exploits
that structure directly; a semantic model generalizes away from it.

Latency was a second, independent objection. Fifty pairs at sequence length
128 measured `547 ms`, and an end-to-end turn measured `753` and `864 ms`
against a `184 ms` P95 baseline and an `800 ms` component budget.

The model asset was removed. `CrossEncoderBackend` and
`load_cross_encoder_backend` remain as the evidence behind this result and as
the worked example of the backend protocol; with the asset absent, requesting
the cross-encoder degrades to the lexical backend rather than disabling
reranking, and that path is covered by tests that need no model file.

## Verification

The shipped configuration was re-measured end to end after all of the above:
TechnicalScore `0.783514`, identical to the sweep. Initialization is
`19580.5 ms`; the 200-session evaluation takes `89.9 s`; reported token usage
is zero. The automated suite passes 965 tests with 7 skipped, and Ruff passes
across `src`, `tests`, and `tools`.

The sealed audit fold, the three-trial release benchmark, and the frozen-input
and delivery-contract checks recorded at `c0d444fa` were **not** re-run for
this runtime.
