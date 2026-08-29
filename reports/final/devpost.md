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
hard constraints and then for phrase adjacency where that helps; and asks only
answerable questions with positive expected conversion gain. Every response is
sanitized to ten unique catalog-valid IDs.

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
- Questions that get better from experience: the policy's response-likelihood
  table was hand-written and unchecked, so the agent now treats it as a prior
  and corrects it from whether shoppers actually answered. It found our two
  worst guesses - `feature` was ranked second-lowest at `0.70` and is really
  the most productive question at `0.964`; `budget` was ranked top tier at
  `0.90` and produced nothing in 17 attempts.
- Reranking that knows when to stay out of the way: the same stage is worth
  `+0.038` HitRate on Browsing and `-0.025` on Buying, where hard constraints
  already make the ranker well informed, so it runs on one route and not the
  other.
- Layered failure containment: advanced components can fail without invalidating
  the Agent contract.
- Auditable evaluation: four development folds select one tagged candidate;
  sealed fold 5 is run once with no post-audit tuning.

## Results

On the unchanged 200-session official public evaluator:

| Metric | Starter | CompassCart |
| --- | ---: | ---: |
| TechnicalScore | 0.106710 | **0.822490** |
| HitRate@10 | 0.125 | 0.9650 |
| MRR | 0.068034 | 0.580968 |
| MTTC | 9.81 | 2.715 |

Scenario HitRate@10 is `0.9000` Boundary, `0.9625` Browsing, `0.9375` Buying,
and `0.9333` Intent Override. Initialization is `19.6 s` and a 200-session run
takes `89.9 s`. Reported token usage is zero, so the estimated API cost for the
800-session private set is USD 0.00. The automated suite passes 1006 tests.

Two ideas were measured and thrown away rather than shipped. Weighting the
rerank by window-local inverse document frequency cost `-0.011`, because the
rerank window is selected by the query and so a local statistic penalizes the
shopper's own terms. A quantized MS MARCO cross-encoder cost `-0.007`: it
produced the best MRR we measured and the worst HitRate, and HitRate carries
the larger weight. Both are written up in `reports/final/rerank-results.md`.

## Challenges

The first diversity implementation compared too many candidates and made broad
Browsing turns take seconds. Restricting diversity to the final list reduced the
same operation to roughly 100 ms. The most important correctness bug was more
subtle: an Intent Override could be parsed as the answer to a question from the
old intent. A one-turn pending attribute plus override-first parsing fixed the
state transition and became a regression test.

The policy memory's first version scored a question by whether the reply grew
the constraint ledger, which measured our parser rather than the shopper: a
requirement stated as free text often parses to nothing yet still reaches
retrieval as query evidence. Under that signal three attributes recorded a
literal zero disclosure rate and the ablation came out at `-0.015`. Scoring the
refusal marker instead - the distinction the simulator actually makes - turned
it into `+0.017`.

The memory also had to learn per route, not in aggregate. Pooling both routes
cost a session; conditioning on the retrieval route recovered it, because the
same question is not the same question on both. `use_case` is answered 44% of
the time on the Buying route and never on Browsing - a shopper still exploring
has not yet formed a view on what the item is for. Nothing in the code knows
that; it came out of 303 observations.

## Team Contributions

1. Product lead: judging strategy, narrative, submission, and demo ownership.
2. Retrieval engineer: catalog index, dense assets, and hybrid fusion.
3. Conversation engineer: parser, ledger, routing, and clarification policy.
4. Quality engineer: evaluator, cross-validation, failure analysis, and tests.
5. Release engineer: offline environment, packaging, checksums, and operations.

Replace these role labels with member names and final contribution details in
the portal. The repository contains the exact commands, reports, and release
checklist used to reproduce the result.
