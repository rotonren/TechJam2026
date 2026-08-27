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
.\.venv\Scripts\python.exe -m tools.run_agent --output results.json
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
.\.venv\Scripts\python.exe -m tools.run_agent --output results-lexical.json
```

Asset corruption, optional dependency failure, or dense inference failure also
switches to lexical retrieval automatically. Neither path performs network I/O.

## Local Verification and Demo

Install the pinned runtime and development dependencies, then run the release
audit against the official 50,000-item catalog:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m tools.release_audit --catalog data\catalog.jsonl
```

The audit verifies the catalog, public-set, evaluator, dependency, and dense
asset fingerprints; exercises an agent response; and builds the allowlisted
submission archive. It fails closed if dense retrieval is unavailable. The
catalog is intentionally not committed; use `tools/download_kit.ps1` to fetch
the organizer kit when it is absent.

For a terminal demo:

```powershell
.\.venv\Scripts\python.exe -m tools.demo_chat --catalog data\catalog.jsonl --top 3
```

For the development-only browser console, provide a random secret of at least
43 characters and start the local server:

```powershell
$env:COMPASSCART_DEBUG_TOKEN = "replace-with-a-random-secret-of-at-least-43-characters"
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m tools.run_debug_server
```

Open `http://127.0.0.1:8765`. The console binds to localhost by default and is
excluded from the submission package.

## Architecture

`CatalogIndex` provides FTS5 BM25, structured attributes, popularity, and a
pure-Python fallback. `SessionStore` keeps a bounded, versioned constraint
ledger and resets obsolete evidence on intent override. `RoutePlanner` selects
Buying, Browsing, or Override weights; `HybridRetriever` fuses lexical,
attribute, profile, and ONNX dense candidates with weighted reciprocal-rank
fusion. `ConstraintRanker` applies hard constraints and final-list diversity.
`QuestionPolicy` asks only when expected conversion gain is positive.
`ResponseBuilder` deduplicates and validates all identifiers.

Additional rationale is in `reports/final/architecture.md`.

## Measured Results

All measurements use the unchanged official evaluator and frozen public data.
The starter baseline scored `0.106710`. The optimized candidate is under owner
review on branch `codex/release-v3-alignment`; its runtime code is commit
`54b2a62`, and the proposed post-review tag is `compasscart-v3-candidate`.

Development CV on folds 1-4 scored `0.662377 +/- 0.036433` (mean
TechnicalScore), with selection score `0.644160`. Fold scores were `0.670738`,
`0.705902`, `0.604819`, and `0.668048`; no runtime fallback occurred. The
maximum fold P95 was `483.890 ms` on the development host.

A fresh Windows dense-runtime reproduction scored `0.660605` on the official
200-session public evaluator, with HitRate@10 `0.840`, MRR `0.376349`, and MTTC
`4.615`. Scenario HitRate@10 was `0.90` Boundary, `0.85` Browsing, `0.8625`
Buying, and `0.733333` Intent Override. Exact dependency versions and data,
evaluator, and catalog hashes are recorded in
`reports/final/final-results.json`.

Fold 5 was viewed during the historical v2 release and is not represented as a
blind audit for this optimized candidate. Version selection used folds 1-4;
the organizer's private 800-session evaluation remains the true blind test.

The agent reports zero prompt and completion tokens. Official runtime API cost
is USD 0.00 per session and USD 0.00 for the full 800-session private set.
Development experiments also used no paid API calls; local asset generation was
a one-time CPU process.

## Limitations

- Intent Override remains the least stable scenario because terse replacement
  messages may expose only one attribute.
- Dense quality depends on text metadata; images are intentionally out of scope.
- Dense ranking boundaries can vary slightly across compatible NumPy, ONNX
  Runtime, and tokenizer versions. Release evidence records exact versions and
  input hashes; `tools.release_audit` fails when the runtime silently falls back
  to lexical retrieval.
- First initialization loads the catalog and ONNX assets, so cold-start memory
  and latency are higher than steady-state response latency. A full-catalog
  dense smoke measured approximately 504 MiB working set after one response
  (576 MiB peak); evaluator processes retain additional trace/session state.
- The default catalog path is relative to the process working directory; pass a
  path to `Agent(...)` when the harness uses a different layout.

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
