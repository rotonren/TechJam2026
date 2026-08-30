# Cross-Session Policy Memory: Results

Recorded on 2026-08-29 against the unchanged official evaluator, the frozen
200-session public set, and the frozen 50,000-product catalog.

## What it does

`QuestionPolicy` weights each candidate clarification by a response
likelihood: how often a shopper can actually answer a question about that
attribute. That table was hand-written and nothing ever checked it.

Every turn supplies the check. The agent asks about one attribute, and the next
message either states a requirement or refuses it. `PolicyMemory` treats the
hand-written value as a Beta prior and updates a posterior from those
observations, so the first session behaves exactly as the hand-written table
did and the estimate moves only as far as evidence carries it.

Observations never touch ground truth. The agent learns whether its own
question was productive, never whether its recommendations were right.

## Ablation

| Configuration | TechnicalScore | Hit@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Memory disabled | 0.783514 | 0.9450 | 0.528714 | 3.380 |
| Memory, pooled across routes | 0.800849 | 0.9400 | 0.560163 | 2.860 |
| **Memory, conditioned on route** | **0.804724** | 0.9450 | 0.561413 | 2.810 |
| Change | **+0.021210** | 0.0000 | +0.032699 | −0.570 |

The disabled arm reproduces the pre-memory runtime exactly, which confirms the
memory is inert when switched off. The gain is better questions converging
faster: MTTC falls by more than half a turn and MRR rises by `0.033`.

Pooling every route into one estimate cost a session. Conditioning on the
retrieval route recovered it and improved MTTC further, for the reason the next
section makes visible.

## What the route split found

The same question is not the same question on both routes:

| Attribute | Browsing | Buying |
| --- | ---: | ---: |
| feature | 53/54 = 0.981 | 55/57 = 0.965 |
| material | 40/52 = 0.769 | 17/24 = 0.708 |
| color | 6/13 = 0.462 | 16/26 = 0.615 |
| use_case | 0/5 = 0.000 | 7/16 = 0.438 |
| style | 0/3 = 0.000 | 4/14 = 0.286 |
| size | 0/2 = 0.000 | 1/4 = 0.250 |
| budget | 0/7 = 0.000 | 0/9 = 0.000 |

`use_case` and `style` are answerable on the Buying route and never answered on
the Browsing route. That is the behaviour you would expect and did not
encode: a shopper who is still exploring has not yet formed a view on what the
item is for or how it should look, while one who opened with a hard requirement
already has. `material` and `feature` pay on both routes because they are what
product text describes.

Nothing in the code knows this. It came out of 303 observations.

## What the pooled estimates found: the catalog is parent-level

The two attributes the memory learned to almost never ask about are the two the
catalog almost never records. Over a 20,000-product sample of the frozen
catalog:

| Attribute | No value in catalog | Learned yield |
| --- | ---: | ---: |
| size | 91.3% | 0.167 |
| budget | 77.6% | 0.000 |
| color | 61.4% | 0.564 |
| use_case | 56.9% | 0.333 |
| style | 56.1% | 0.235 |
| material | 31.8% | 0.750 |
| feature | 57.7% | **0.973** |

A `parent_asin` is a parent product, not a colour or size SKU variant, so size
is not a parent-level property at all and is absent from nine products in ten.
A question about it cannot be answered however it is phrased. The agent was not
told this; it inferred it from whether its own questions were answered.

`feature` is the one attribute that breaks the pattern - 57.7% absent yet
answered 97.3% of the time - and for a different reason. It is the fallback
bucket of the evaluator's constraint classifier, so it absorbs every stated
requirement that does not match another attribute's keywords. Two mechanisms
are therefore visible in one table: catalogue sparsity sets the ceiling for
most attributes, and classifier bucketing lifts one of them above it.

## Why none of this is fitted to the public targets

The memory stores eight attribute-level rates. It holds no product
identifiers, no session identifiers and no labels, and the organizer has
confirmed zero target-product overlap between the public and private splits, so
there is nothing here that could be a memorized answer. The same is true of the
rerank stage, which has no learned parameters at all, and of the strategy
selector, which is a deterministic rule. What transfers is the mechanism, not
the sample.

## Per-scenario effect

| Scenario | Before | After | HitRate | MTTC |
| --- | --- | --- | ---: | ---: |
| Browsing | 0.9500 / 3.625 | 0.9625 / 2.688 | +0.0125 | −0.937 |
| Buying | 0.9375 / 2.700 | 0.9375 / 2.212 | 0.0000 | −0.488 |
| Boundary | 0.9000 / 3.800 | 0.9000 / 3.300 | 0.0000 | −0.500 |
| Intent Override | 0.9667 / 4.400 | 0.9333 / 4.567 | **−0.0334** | +0.167 |

Intent Override still pays for this. Route conditioning recovered it from
`0.9000` to `0.9333` but not to the `0.9667` it had before the memory existed:
override sessions route as Buying and are pooled with ordinary Buying turns,
even though their question sequence restarts mid-conversation. Conditioning on
override state as well as route is the obvious next refinement and is not done.

## What the agent corrected

After 303 observations across 200 sessions, pooled over both routes:

| Attribute | Asked | Disclosed | Observed rate | Hand-written prior | Posterior | Shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| feature | 111 | 108 | 0.973 | 0.70 | 0.9546 | **+0.2546** |
| other | 17 | 13 | 0.765 | 0.75 | 0.7600 | +0.0100 |
| material | 76 | 57 | 0.750 | 0.90 | 0.7643 | −0.1357 |
| color | 39 | 22 | 0.564 | 0.90 | 0.6213 | −0.2787 |
| use_case | 21 | 7 | 0.333 | 0.85 | 0.4759 | −0.3741 |
| style | 17 | 4 | 0.235 | 0.80 | 0.4160 | −0.3840 |
| size | 6 | 1 | 0.167 | 0.85 | 0.5571 | −0.2929 |
| budget | 16 | 0 | 0.000 | 0.90 | 0.3000 | −0.6000 |

The two largest corrections are the two the hand-written table got backwards.
`feature` was ranked second-lowest at `0.70` and is in fact the most productive
question at `0.973`; `budget` was ranked in the top tier at `0.90` and produced
nothing in 16 attempts. The table was also uniformly overconfident: six of
eight attributes moved down.

## What the numbers do not show

There is no learning curve. Split by evaluation order, the enabled arm scores
`0.804`, `0.796`, `0.827`, `0.791` across the four quartiles against `0.795`,
`0.748`, `0.801`, `0.789` disabled — higher in every quartile, including the
first, but with no upward slope.

That is the honest reading, and it has a mechanical explanation: 303
observations spread over eight attributes, against a prior strength of `8.0`,
converge inside roughly the first fifty sessions. The benefit arrives early
rather than accumulating. Quartile variance in both arms is driven by sample
difficulty, not by learning, so a rising curve should not be claimed from this
data.

## Degradation

The memory can improve behaviour and can never be required for it:

- `evolution_enabled=False` or `COMPASSCART_DISABLE_EVOLUTION=1` returns the
  hand-written prior for every attribute.
- A harness that constructs one agent per session never accumulates evidence,
  so every session sees the prior — the pre-memory behaviour, not a degraded
  one.
- An exception inside `observe` disables learning for the rest of the process
  rather than leaving a partially updated estimate.
- A shopper segment only overrides the global estimate after twelve
  observations of its own; below that it is noise, not personalization. The
  segment table is bounded at 256 entries.

## A defect this work exposed

The first implementation scored the question by whether the reply grew the
constraint ledger. That measured the parser, not the shopper: a requirement
stated as free text — "tagless collar" — often parses to no structured
constraint while still reaching retrieval and the rerank stage as query
evidence. Under that signal `budget`, `size` and `use_case` each recorded a
literal zero disclosure rate, the memory learned nonsense, and the ablation
measured `-0.015`. Switching the signal to the refusal marker, which is what
the simulator actually distinguishes, produced the table above and `+0.017`.

# Per-Turn Strategy Selection: Results

The pipeline ran the same shape every turn regardless of whether the previous
turn made progress. `StrategySelector` names what this turn should do instead,
deterministically, from the distilled negative evidence on the session.

## Ablation

| Configuration | TechnicalScore | Hit@10 | MRR | MTTC | Strategy mix |
| --- | ---: | ---: | ---: | ---: | --- |
| Selector disabled | 0.804724 | 0.9450 | 0.561413 | 2.810 | probe 551 |
| **`open_probe` + `exploit`** | **0.822490** | **0.9650** | **0.580968** | **2.715** | probe 508, open_probe 11, exploit 17 |
| All three, stall limit 3 | 0.802203 | 0.9400 | 0.563343 | 2.840 | probe 480, relax 53, open_probe 5 |
| All three, stall limit 2 | 0.801661 | 0.9400 | 0.560871 | 2.830 | probe 463, relax 72, open_probe 2 |
| All three, stall limit 1 | 0.794374 | 0.9300 | 0.557913 | 2.900 | probe 401, relax 146, open_probe 1 |

## `relax`, and why it was removed

`relax` demoted the weakest hard constraint to a preference whenever the
conversation stalled, on the theory that a stall means the filters are too
tight. It cost score monotonically in how often it fired: `-0.003` at 53
firings, `-0.003` at 72, `-0.010` at 146.

A stall in this benchmark does not mean the filters are wrong. It means the
target is hard to find, and the hard constraints are the most reliable evidence
available - the simulated shopper quotes them from the target product's own
text. Widening them discards the best signal in the session.

This is the same result the rerank stage produced on the Buying route, arrived
at independently: **when the constraints are reliable, second-guessing them is
destructive.**

Removing `relax` did more than undo its cost. The score moved from `-0.003` to
`+0.018`, because `relax` had been intercepting turns that belonged to
`open_probe`: open questions rose from 2 firings to 11, and HitRate@10 went to
`0.9650`, the highest measured in this work.

## Why eleven turns are worth `+0.018`

`open_probe` fires only when the structured question policy has nothing worth
asking. Those are exactly the turns that would otherwise be spent: the
simulated shopper replies "Those options are not quite right yet. Ask me about
one specific attribute" and discloses nothing at all. Eleven wasted turns
converted into eleven turns that collect a requirement is a large effect
concentrated in a small number of sessions, which is why firing count is a poor
proxy for value here - and why the first measurement, where `relax` left
`open_probe` only two turns, looked like noise.

## What generalizes

Two conditioning signals were tested across four independent experiments:

| Condition | Where | Result |
| --- | --- | ---: |
| Route - what kind of conversation this is | rerank weight | +0.022 |
| Route | question priors | +0.021 |
| Structured question has no value | strategy selector | +0.018 |
| Stall - whether the last turn failed | strategy selector | −0.003 to −0.010 |

Adaptation is not valuable in itself. It pays when it is conditioned on a real
structural difference - a Browsing turn genuinely is not a Buying turn, and a
turn with no answerable question genuinely is not a turn with one. "The last
turn did not help" is not such a difference, and conditioning on it lost score
in every configuration tried.

# End-to-End Verification

Both arms measured in one process against the frozen public set:

| Configuration | TechnicalScore | Hit@10 | MRR | MTTC | Init | Eval | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All three layers off | 0.761209 | 0.9300 | 0.489030 | 3.525 | 19459.7 ms | 89.9 s | 0 |
| Shipped defaults | 0.822490 | 0.9650 | 0.580968 | 2.715 | 19415.7 ms | 72.7 s | 0 |

Disabling the rerank stage, the policy memory and the strategy selector
reproduces `0.761209` exactly, so each layer is inert when switched off.

| Scenario | All off | Shipped | HitRate | MTTC |
| --- | --- | --- | ---: | ---: |
| Boundary | 0.9000 / 5.200 | 1.0000 / 3.000 | +0.1000 | −2.200 |
| Browsing | 0.9125 / 3.812 | 0.9750 / 2.650 | +0.0625 | −1.162 |
| Buying | 0.9375 / 2.700 | 0.9625 / 2.050 | +0.0250 | −0.650 |
| Intent Override | 0.9667 / 4.400 | 0.9333 / 4.567 | **−0.0334** | +0.167 |

Boundary's `1.0000` is ten sessions out of ten and should not be read as a
reliable rate. Intent Override is the one scenario that regresses, by a single
session.

The process resident-set figures from this run (668 MiB and 947 MiB) are not a
clean comparison: the script holds the evaluator's own copy of the catalog and
keeps both agents alive, so the second figure includes the first. A dedicated
memory measurement has not been run for this runtime.
