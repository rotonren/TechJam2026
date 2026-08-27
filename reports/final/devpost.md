# CompassCart: An Offline Shopping Agent That Handles Changed Minds

## Inspiration

Conversational shopping fails when a system treats every turn as a new keyword
query or keeps following a preference the shopper has already replaced. Track 4
rewards something closer to real assistance: find the exact item early, ask
useful questions, and recover immediately when intent changes.

## What It Does

CompassCart turns each conversation into a bounded, versioned constraint ledger.
It routes precise purchases, broad browsing, and intent overrides differently;
fuses lexical, attribute, profile, and local semantic retrieval; reranks against
hard constraints; and asks only answerable questions with positive expected
conversion gain. Every response is sanitized to ten unique catalog-valid IDs.

## How We Built It

The lexical core uses SQLite FTS5 BM25 with normalized catalog attributes and a
pure-Python fallback. A quantized `all-MiniLM-L6-v2` ONNX encoder and int8
catalog vectors add semantic recall without a service or vector database.
Weighted reciprocal-rank fusion combines independent candidate sources. The
dialog layer tracks the source, confidence, turn, version, hardness, and status
of each constraint, while retaining only four evidence-bearing messages.

The clarification policy estimates how much a candidate set would split on an
attribute, discounts late or repeated questions, and excludes fields the user
cannot reliably answer. Override parsing runs before pending-question parsing,
so "Actually, I need leather" cannot accidentally become a category answer.

## What Makes It Different

- Offline-first by design: zero API calls, tokens, credentials, or network I/O.
- Real override semantics: obsolete constraints and free text are superseded,
  not merely down-weighted.
- Conversion-aware questions: the agent values the next turn, not question count.
- Layered failure containment: advanced components can fail without invalidating
  the Agent contract.
- Auditable evaluation: four development folds select one candidate, input and
  dependency fingerprints are recorded, and the private 800 sessions remain
  the true blind test.

## Results

On the unchanged 200-session official public evaluator, the optimized
owner-review candidate achieved:

| Metric | Starter | CompassCart |
| --- | ---: | ---: |
| TechnicalScore | 0.106710 | 0.660605 |
| HitRate@10 | 0.125 | 0.840 |
| MRR | 0.068034 | 0.376349 |
| MTTC | 9.81 | 4.615 |

Development folds 1-4 averaged `0.662377 +/- 0.036433` (selection score
`0.644160`), with fold scores `0.670738`, `0.705902`, `0.604819`, and
`0.668048`. All four folds completed without runtime fallback; maximum P95 was
`483.890 ms`. Exact public metrics and reproducibility fingerprints are in
`final-results.json`. Estimated API cost for the 800-session private set is USD
0.00. Scenario HitRate@10 was `0.90` Boundary, `0.85` Browsing, `0.8625`
Buying, and `0.733333` Intent Override.

Fold 5 was already viewed during the historical v2 release and was not used as
a blind selection gate for this optimized candidate.

## Challenges

The first diversity implementation compared too many candidates and made broad
Browsing turns take seconds. Restricting diversity to the final list reduced the
same operation to roughly 100 ms. The most important correctness bug was more
subtle: an Intent Override could be parsed as the answer to a question from the
old intent. A one-turn pending attribute plus override-first parsing fixed the
state transition and became a regression test.

Intent Override remains the hardest scenario. Terse replacement messages may
reveal only one attribute, so future work should improve local query expansion
without adding a network dependency or leaking old-intent evidence.

## Team Contributions

1. Product lead: judging strategy, narrative, submission, and demo ownership.
2. Retrieval engineer: catalog index, dense assets, and hybrid fusion.
3. Conversation engineer: parser, ledger, routing, and clarification policy.
4. Quality engineer: evaluator, cross-validation, failure analysis, and tests.
5. Release engineer: offline environment, packaging, checksums, and operations.

Replace these role labels with member names and final contribution details in
the portal. The repository contains the exact commands, reports, and release
checklist used to reproduce the result.
