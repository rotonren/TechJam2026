# How the Design Arrived Here

Written for the team. Every number below comes from the unchanged official
evaluator over the frozen 200-session public set. The point of this document is
not the final score; it is the order the decisions were made in, and what each
measurement ruled out.

## The lineage

```
0.761209  ─ starting point (4f56f2f)
     │
     │  oracle probe: retrieval is not the bottleneck
     ▼
0.771831  ─ phrase-adjacency rerank, one weight for every route      +0.0106
     │
     │  the same stage helps Browsing and hurts Buying
     ▼
0.783514  ─ rerank on Browsing only                                  +0.0223
     │
     │  the question priors were never checked against anything
     ▼
0.800849  ─ cross-session policy memory, pooled                      +0.0173
     │
     │  the same question is not the same question on both routes
     ▼
0.804724  ─ policy memory conditioned on route                       +0.0212
     │
     │  a turn with no worthwhile question is a wasted turn
     ▼
0.822490  ─ per-turn strategy selection            SHIPPED DEFAULT   +0.0613
     │
     └─── optional, needs credentials ───▶ 0.826831  LLM rerank      +0.0043
```

Rejected along the way, each measured rather than argued:

```
window-local IDF weighting             -0.011     structurally wrong
ONNX cross-encoder backend             -0.007     confounded, see below
`relax` strategy                       -0.010     conditioned on the wrong thing
constraint-count disclosure signal     -0.015     measured the parser, not the shopper
grouped model prompt                   -0.003     labels were inverted, then still worse
state-selected prompt                  +0.0002    gains did not compose
layered catalog discovery              ±0.000     and cost 92.7s of startup
```

## Step 1 — Measure the ceiling before building anything

The first thing built was not a feature. It was an oracle probe that
instrumented `HybridRetriever` and `ConstraintRanker` to record where the target
product sat at each stage, across all 200 sessions and 691 turns.

| Recall of the target in the retrieval output | @10 | @20 | @50 | @100 | any |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 200 sessions | 0.840 | 0.925 | 0.960 | 0.980 | 0.990 |

Two sessions out of two hundred never retrieved the target at all. The target
was inside the final top-10 in 186 sessions but at rank 1 in only 64.

**Retrieval was not the problem. Ordering was.** That single table decided the
next three days: no work went into recall, and every accepted change targeted
position.

## Step 2 — Add the signal the pipeline could not express

Fusion and constraint scoring both work at term level. A requirement stated as
a contiguous phrase scores no better against a product whose title contains
that exact phrase than against one mentioning the same words far apart.

Phrase adjacency closed that gap: `+0.0106`.

## Step 3 — Notice that one stage was doing two opposite things

Per scenario, the same stage measured:

| Scenario | Hit@10 | MRR |
| --- | ---: | ---: |
| Browsing | **+0.0375** | **+0.1006** |
| Buying | −0.0250 | −0.0180 |

A Buying turn already carries explicit hard constraints, so the constraint
ranker is well informed and reordering it destroys good work. Turning the stage
off on Buying was worth more than every weight and window sweep combined:
`0.771831 → 0.783514`.

This is the first appearance of the pattern that held for the rest of the work:
**adaptation pays when it is conditioned on a real structural difference.**

## Step 4 — Check a table nobody had ever checked

`QuestionPolicy` weighted each candidate clarification by a hand-written
response likelihood. Every turn supplies the evidence to check it: the agent
asks about an attribute, and the reply either states a requirement or refuses.

Treating the hand-written value as a prior and updating a posterior was worth
`+0.0173`, and it corrected the two entries we had backwards:

| Attribute | Our guess | Observed | Why |
| --- | ---: | ---: | --- |
| `feature` | 0.70 | **0.973** | It is the classifier's fallback bucket, so it absorbs everything |
| `budget` | 0.90 | **0.000** | 77.6% of products record no price at parent level |
| `size` | 0.85 | 0.167 | 91.3% record no size - a `parent_asin` is not a SKU variant |

The agent inferred a structural fact about the catalog from nothing but which
of its own questions got answered.

## Step 5 — Condition the memory too

Pooling both routes into one estimate cost a session. Splitting by route
recovered it and improved MTTC further, because the same question genuinely is
not the same question:

| Attribute | Browsing | Buying |
| --- | ---: | ---: |
| `use_case` | 0/5 = 0.000 | 7/16 = 0.438 |
| `style` | 0/3 = 0.000 | 4/14 = 0.286 |

A shopper still exploring has not yet formed a view on what the item is for.
One who opened with a hard requirement already has.

## Step 6 — Spend the turns that were being wasted

When the question policy has nothing worth asking, the turn is spent for
nothing: the shopper answers "ask me about one specific attribute" and
discloses zero. Asking an open question instead fires eleven times in 536 turns
and is worth `+0.0178`.

A third strategy, `relax`, dropped the weakest hard constraint whenever the
conversation stalled. It lost score monotonically in how often it fired -
`-0.003` at 53 firings, `-0.010` at 146 - because a stall does not mean the
filters are wrong. It means the target is hard to find, and the hard
constraints are quoted from the product itself.

**Conditioned on route: +0.022, +0.021. Conditioned on "did the last turn
fail": negative every time.**

## Step 7 — The LLM, and three wrong conclusions

Pillar I of the brief names LLM semantic ranking, so it was implemented and
measured rather than assumed. The path was not straight.

**The first measurement said `-0.001266`, and it was wrong.** The LLM
configuration used a rerank window of 20 while the baseline used 50, and the
window was a stage-level parameter, so lowering it for the model also lowered
it for the phrase backend on Browsing. An isolation run with no model at all
measured window 20 at `0.812999` against window 50 at `0.822490`. The window
alone cost `-0.0095`, and that had been charged to the model. Making the window
per route turned `-0.001266` into `+0.004341`.

**The explanation offered for it was also wrong.** Boundary was said to suffer
because those sessions route as Buying and so met the model. Route counts show
Boundary is routed Browsing on all thirty of its turns; the model never ran on
it.

**The structured prompt inverted a label.** Grouping the ledger by role should
have been strictly better than a token bag - which cannot distinguish a hard
requirement from a preference, renders a budget of `80.00` as the tokens `80`
and `00`, and makes a negative constraint read exactly like a positive one. It
measured worse, because an override's replacement requirement was presented as
`Preferences (helpful, not required)` on the very turn the shopper said it was
what they needed. `is_hard=False` on that constraint means *this free text
cannot safely become a retrieval filter*, not *the shopper is indifferent*.
Grouping by `source` instead recovered `+0.0025`.

The cross-encoder's `-0.007` carries the same window confound and has not been
re-measured. It is reported as measured, not as settled.

## Step 8 — Where the layers stopped being independent

The two prompt forms win on different turns, so taking each where it wins
should have composed to about `0.832`. It scored `0.827009` - one part in five
thousand above the flat prompt, which is not a result.

It also lost `0.067` of Boundary MRR, on a scenario the model never runs on.
Boundary is served by the phrase backend and nothing about it changed - except
that reordering Buying candidates changes which clarification gets asked, which
changes what the cross-session memory learns, which reaches every later
session.

**The three layers are coupled through the shared memory. A configuration that
is locally optimal per route is not the sum of the per-route optima.** That is
the most useful thing this branch produced, and it explains several earlier
surprises where changing one route moved a scenario it had no contact with.

## Step 9 — Check whether any of it generalizes

Everything above was read off the same 200 sessions it was tuned on. So one
fold of 40 was sealed at the start of the round and opened once, at the end,
after the configuration was frozen.

| | TechnicalScore |
| --- | ---: |
| Tuning folds 1-4, mean | 0.812211 |
| All 200 sessions | 0.822490 |
| **Fold 5, sealed** | **0.834997** |

The sealed fold scored above both. Separately, disabling the learning layer
under the same folds costs `-0.019468`, so the memory works on 40 sessions and
is not living off the accumulated public set.

Neither result certifies a `+0.004` decision - the tuning folds alone span
0.769 to 0.837. They certify that the `+0.061` of accepted change did not come
from memorizing the public set. Full tables in `validation-evidence.md`.

## What generalizes

**Measure the ceiling first.** The oracle probe took thirty minutes and decided
three days of work. Without it, the obvious move would have been to improve
recall, which was already at 99%.

**Isolate one variable.** Violating this produced a confidently wrong
conclusion that stood for several hours and would have gone into the report.
The controls that caught it cost 90 seconds each.

**Adaptation needs a real structural difference to condition on.** Route works.
"The last turn failed" does not, in four configurations.

**A field's meaning does not travel with its name.** `is_hard` means "safe to
filter on" in the retrieval path. Read as "important to the shopper" in a new
path, it inverted what the prompt said, and cost `0.105` Override MRR.

**Structure is only better than a bag of words while its labels are right.** A
flat query is uninformative; a mislabelled structured one is confidently wrong,
which is worse.

**Measure the thing, not a proxy for it.** Three separate mistakes in this work
came from counting net active constraints as a stand-in for "did the shopper
tell us something" - it counts the parser, not the shopper, and deduplication
and supersession both make it lie.

## What was deliberately not done

- **Recall work.** 99% of targets are already in the pool.
- **Shipping the LLM by default.** `submission_rules.md` forbids shipping API
  keys and states official scoring may disable network access, so the offline
  path is the one that will run. Both numbers are reported.
- **Deleting the rejected code.** The cross-encoder and LLM backends stay,
  disabled and tested, because they are the evidence behind the report's
  claims. A reviewer can flip one config value and reproduce any row.
- **Catalog layer discovery.** It produced a parser vocabulary identical to the
  cheap path for 92.7 seconds of startup and 77.4 MiB, so it is opt-in.
