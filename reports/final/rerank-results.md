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

## Rejected: hosted LLM listwise reranking

`LlmRerankBackend` sends the rerank window to an OpenAI-compatible endpoint and
asks for a full ordering. Measured with `deepseek-chat`, applied to the Buying
route while Browsing kept the phrase backend:

| Configuration | TechnicalScore | Hit@10 | MRR | MTTC | Prompt tokens | Completion | Wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Phrase, Browsing only** | **0.822490** | **0.9650** | 0.580968 | 2.715 | 0 | 0 | 73 s |
| LLM on Buying, weight 0.8 | 0.821224 | 0.9600 | 0.582079 | 2.670 | 359,732 | 16,827 | 349 s |
| LLM on Buying, weight 0.3 | 0.814371 | 0.9500 | 0.585236 | 2.810 | 387,959 | 18,162 | 367 s |

Both weights lose, and the lighter blend loses more. A single evaluation costs
about 360,000 prompt tokens and runs 4.8 times slower.

*Caveat: the LLM configurations use `rerank_window=20` rather than the default
`50`, because a 50-candidate window is roughly 3,000 prompt tokens per call.
The Browsing arm is therefore not strictly comparable. The policy memory also
couples the routes - changing Buying's candidate order changes which questions
are asked, which changes what the memory learns, which reaches Browsing.*

### Where it went wrong

Per scenario, against the phrase-only default:

| Scenario | Hit@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Buying | 0.9625 → 0.9750 | 0.506 → 0.461 | 2.050 → 1.762 |
| Browsing | 0.9750 → 0.9625 | 0.635 → 0.690 | 2.650 → 2.712 |
| Intent Override | 0.9333 → 0.9333 | 0.661 → 0.693 | 4.567 → 4.600 |
| **Boundary** | **1.0000 → 0.9000** | 0.505 → 0.349 | 3.000 → 3.800 |

On Buying itself the model helps coarse relevance and hurts fine ordering: it
puts the target inside the top ten more often and half a turn earlier, while
placing it worse within that ten. Asking for a permutation of twenty items
scrambles an ordering that was already good.

The losses concentrate in Boundary, which falls from ten hits out of ten to
nine. Boundary sessions route as Buying on specificity, so the model ran on
them - and a shopper saying "I don't have a preference for colour, use your
judgment" is precisely the case where a ranking model has no evidence to act
on and reorders anyway.

### Why the lexical backend keeps winning

This is the second model-based reranker to lose to phrase matching here, after
the ONNX cross-encoder at `-0.007`. Both fail for the same structural reason,
which the organizer confirmed in the Track 4 Q&A: private intent cards are
derived from the same catalog metadata records exposed to participants. The
simulated shopper quotes the target product's own text. Exact phrase
containment is not a proxy for the signal - it *is* the signal, and a model
whose strength is generalizing past surface form generalizes past it.

The backend is kept, disabled by default. It needs no dependency beyond the
standard library, reads credentials only from the environment, validates that a
reply is a true permutation of the window before trusting it, and falls back to
the lexical backend on a timeout, a refusal, an unparseable reply, or absent
credentials - which is the expected case under `submission_rules.md`, since
official scoring may disable network access and forbids shipping keys.
