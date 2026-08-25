# Three-Minute Demo Script

## Setup (0:00-0:20)

Show the terminal with `data/catalog.jsonl` present and no API-key environment
variables. State that all recommendations come from the frozen 50,000-product
catalog and the agent reports zero tokens.

## Browsing Session (0:20-1:10)

Reset a session with an anonymized preference profile. Begin with a vague apparel
request. Show the first ranked recommendations and the structured clarification
attribute. Answer with a material or use-case preference, then show the updated
top ten and the earlier hit turn. Point to the constraint ledger trace: only
bounded structured evidence is retained.

## Intent Override Session (1:10-2:00)

Begin with one product intent, answer one clarification, then say: "Actually,
ignore my earlier preference. What I need is a black leather belt." Show that
the intent version increments, obsolete constraints are superseded, the pending
question is not applied to the override message, and the result list changes
immediately.

## Offline Proof (2:00-2:35)

Disable network access at the host level, start a fresh process, and repeat the
override response. Then set `COMPASSCART_DISABLE_DENSE=1` and show that the same
Agent contract still returns valid catalog IDs through lexical fallback.

## Evidence (2:35-3:00)

Close on the official metrics: starter `0.106710`, selected development CV mean
`0.424038`, selection score `0.405810`, zero fallbacks across folds 1-4, zero
runtime API cost, and deterministic package SHA256. Do not claim the sealed
fold-5 or final full-set number until the release evidence file is generated.
