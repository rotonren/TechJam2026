# CompassCart

CompassCart is an offline-first conversational product-search agent for TikTok
TechJam 2026 Track 4. It combines lexical and local dense retrieval with a
versioned constraint ledger, intent-override handling, route-aware ranking,
and a conversion-gain clarification policy. Official scoring requires no
network connection, API key, hosted model, or external database.

## Reproduction

The release was developed and tested with CPython 3.12.13 on Windows 11. The
runtime supports Python 3.10 or newer. From the repository or extracted
submission directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux and macOS use `.venv/bin/python` in place of the Windows executable.
Place the organizer-provided frozen catalog at `data/catalog.jsonl`. The
submission does not redistribute public labels or evaluator code.

Run the official public harness from a participant-kit checkout:

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m tools.run_agent `
  --catalog data/catalog.jsonl `
  --dataset data/public_set.jsonl `
  --output results.json `
  --evidence-output results-evidence.json
```

The entry point is `agent.Agent`. It implements `reset(session_id,
user_profile)` and `respond(session_id, user_message, turn, top_k)` exactly as
defined by the official contract.

## Runtime Assets

The submission includes a quantized `all-MiniLM-L6-v2` ONNX encoder and int8
catalog vectors. `assets/SHA256SUMS` authenticates every shipped dense asset.
Set `COMPASSCART_DISABLE_DENSE=1` to force the deterministic lexical fallback:

```powershell
$env:COMPASSCART_DISABLE_DENSE = "1"
.\.venv\Scripts\python.exe -m tools.run_agent `
  --catalog data/catalog.jsonl `
  --dataset data/public_set.jsonl `
  --output results-lexical.json `
  --evidence-output results-lexical-evidence.json
```

Asset corruption, optional dependency failure, or dense inference failure also
switches to lexical retrieval automatically. Neither path performs network I/O.

The runtime resolves its bundled dense assets from the installed package rather
than the process working directory. It can therefore be imported from an
arbitrary CWD after extracting the submission ZIP; pass the catalog path
provided by the harness when the catalog lives outside that extraction.

### Release Benchmark (Non-competition)

The release benchmark is an engineering validation tool, not a competition
runtime dependency. It additionally requires `psutil` for process-memory
measurement and uses the development-only frozen transcript and catalog:

```powershell
.\.venv\Scripts\python.exe -m tools.benchmark_release `
  --catalog data/catalog.jsonl `
  --transcript var/balanced-hardening/benchmark-transcript.jsonl `
  --trials 3 --cwd-mode outside `
  --output var/balanced-hardening/benchmark-p0.json `
  --compare var/balanced-hardening/benchmark-r0-wall-v2.json
```

## Architecture

`CatalogIndex` provides FTS5 BM25, structured attributes, popularity, and a
pure-Python fallback. `SessionStore` keeps a bounded, versioned constraint
ledger and resets obsolete evidence on intent override. `RoutePlanner` selects
Buying, Browsing, or Override weights; `HybridRetriever` fuses lexical,
attribute, profile, and ONNX dense candidates with weighted reciprocal-rank
fusion. `ConstraintRanker` applies hard constraints and final-list diversity.
`RerankStage` then rescores the head of that list for phrase adjacency, which
term-level scoring cannot see, and applies it on the Browsing route only.
`QuestionPolicy` asks only when expected conversion gain is positive, and
weights each candidate question by a `PolicyMemory` estimate that starts at our
hand-written prior and is corrected by whether the shopper answered previous
questions. `ResponseBuilder` deduplicates and validates all identifiers.

Additional rationale is in `reports/final/architecture.md`; the rerank
experiments, including the two that were measured and rejected, are in
`reports/final/rerank-results.md`.

## Measured Results

All measurements use the unchanged official evaluator and frozen public data.
The starter baseline scored `0.106710`. The current runtime scores `0.800849`
on the official 200-sample public evaluation (HitRate@10 `0.9400`, MRR
`0.560163`, MTTC `2.860`, efficiency `0.8140`).

| Stage | TechnicalScore | Change |
| --- | ---: | ---: |
| Team result before this round | 0.761209 | — |
| Phrase rerank, one weight for every route | 0.771831 | +0.010622 |
| Phrase rerank, Browsing route only | 0.783514 | +0.022305 |
| Cross-session policy memory | 0.800849 | +0.039640 |

Each stage has an ablation switch, and disabling both reproduces `0.761209`
exactly.

Initialization is `19580.5 ms`, down from roughly `132 s`, because catalog
layer discovery is now opt-in; see `docs/attribute_schema.md`. Two rerank
variants were measured and rejected rather than kept: window-local IDF
weighting at `-0.011`, and an ONNX cross-encoder backend at `-0.007`. Both,
with their per-scenario numbers, are recorded in
`reports/final/rerank-results.md`; the policy memory's ablation and the
corrections it made to our hand-written question priors are in
`reports/final/evolution-results.md`.

The full automated suite passes 990 tests with 7 skipped, and Ruff lint passes
across `src`, `tests`, and `tools`.

Earlier stable evidence remains in `reports/final/final-results.json` and
`reports/final/score-results-c0d444fa-2026-08-27.json`, recorded at commit
`c0d444fa` when the public score was `0.660411`. The sealed audit fold, the
three-trial resource benchmark, and the frozen-input and delivery-contract
checks in those files were not re-run for the current runtime; only the public
evaluation, the automated suite, and the initialization measurement above were.
macOS and Linux verification are pending.

The agent reports zero prompt and completion tokens. Official runtime API cost
is USD 0.00 per session and USD 0.00 for the full 800-session private set.
Development experiments also used no paid API calls; local asset generation was
a one-time CPU process.

## Limitations

- The policy memory trades Intent Override for Browsing: it costs that
  scenario two hits, `0.9667` to `0.9000`, while lifting Browsing to `0.9625`
  and cutting MTTC from `3.380` to `2.860`. The memory is not route-aware
  the way the rerank stage is, which is the obvious next step.
- Boundary is the weakest scenario at `0.9000` HitRate@10 on only ten public
  sessions, so its measurement is noisy.
- The rerank stage is disabled on the Buying route, which is where its phrase
  evidence measured net harmful. Intent Override sessions route as Buying, so
  they do not benefit from it either.
- Override payload extraction recognizes a family of replacement lead-ins and
  otherwise falls back to the message's final sentence. A paraphrase that
  states the new requirement somewhere else would still be missed.
- Dense quality depends on text metadata; images are intentionally out of scope.
- First initialization loads the catalog and ONNX assets, so cold-start memory
  and latency are higher than steady-state response latency. A full-catalog
  dense smoke measured approximately 504 MiB working set after one response
  (576 MiB peak); evaluator processes retain additional trace/session state.
- The catalog is organizer-provided and is not included in the submission ZIP;
  pass its path to `Agent(...)` when the harness uses a different layout.

## Team Responsibilities

The five-person team can map names to these release roles before submission:

1. Product lead: judging alignment, scope, Devpost narrative, and final demo.
2. Retrieval engineer: catalog indexing, dense assets, and hybrid fusion.
3. Conversation engineer: parser, state ledger, routing, and clarification.
4. Quality engineer: evaluator, CV discipline, failure analysis, and tests.
5. Release engineer: offline deployment, packaging, checksums, and operations.

## Attribution

Amazon Reviews 2023 data terms and attribution are documented in
`DATA_ATTRIBUTION.md`. The local embedding model notice and Apache-2.0 text are
in `MODEL_ATTRIBUTION.md` and `licenses/`. CompassCart source code and
generated competition artifacts remain subject to the team's chosen submission
license.
