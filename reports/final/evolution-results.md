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
| **Memory enabled** | **0.800849** | 0.9400 | 0.560163 | 2.860 |
| Change | **+0.017335** | −0.0050 | +0.031449 | −0.520 |

The disabled arm reproduces the pre-memory runtime exactly, which confirms the
memory is inert when switched off. The gain is better questions converging
faster: MTTC falls by half a turn and MRR rises by `0.031`, against one lost
session.

## What the agent corrected

After 310 observations across 200 sessions:

| Attribute | Asked | Disclosed | Observed rate | Hand-written prior | Posterior | Shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| feature | 111 | 107 | 0.964 | 0.70 | 0.9462 | **+0.2462** |
| other | 16 | 12 | 0.750 | 0.75 | 0.7500 | +0.0000 |
| material | 78 | 58 | 0.744 | 0.90 | 0.7581 | −0.1419 |
| color | 40 | 22 | 0.550 | 0.90 | 0.6083 | −0.2917 |
| use_case | 23 | 7 | 0.304 | 0.85 | 0.4452 | −0.4048 |
| style | 19 | 4 | 0.211 | 0.80 | 0.3852 | −0.4148 |
| size | 6 | 1 | 0.167 | 0.85 | 0.5571 | −0.2929 |
| budget | 17 | 0 | 0.000 | 0.90 | 0.2880 | −0.6120 |

The two largest corrections are the two the hand-written table got backwards.
`feature` was ranked second-lowest at `0.70` and is in fact the most productive
question at `0.964`; `budget` was ranked in the top tier at `0.90` and produced
nothing in 17 attempts. The table was also uniformly overconfident: six of
eight attributes moved down.

## What the numbers do not show

There is no learning curve. Split by evaluation order, the enabled arm scores
`0.804`, `0.783`, `0.825`, `0.791` across the four quartiles — higher than the
disabled arm in every quartile, including the first, but with no upward slope.

That is the honest reading, and it has a mechanical explanation: 310
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
