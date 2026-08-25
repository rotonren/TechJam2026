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
The starter baseline scored `0.106710`. CompassCart development CV on sealed
folds 1-4 scored `0.424038 +/- 0.036455` (mean TechnicalScore), with selection
score `0.405810`. Fold scores were `0.444902`, `0.362723`, `0.431955`, and
`0.456572`; no runtime fallback occurred. Per-fold P95 response latency ranged
from 211 ms to 416 ms on the development host.

The once-only sealed fold 5 scored `0.370375` with no fallback. The final
unchanged official 200-session public evaluation scored `0.413305`, with
HitRate@10 `0.52`, MRR `0.231685`, and MTTC `6.81`. Scenario HitRate@10 was
`0.60` Boundary, `0.5375` Browsing, `0.575` Buying, and `0.30` Intent Override.

The agent reports zero prompt and completion tokens. Official runtime API cost
is USD 0.00 per session and USD 0.00 for the full 800-session private set.
Development experiments also used no paid API calls; local asset generation was
a one-time CPU process.

## Limitations

- Intent Override remains the least stable scenario because terse replacement
  messages may expose only one attribute.
- Dense quality depends on text metadata; images are intentionally out of scope.
- First initialization loads the catalog and ONNX assets, so cold-start memory
  and latency are higher than steady-state response latency.
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
`DATA_ATTRIBUTION.md`. The local embedding model notice is in
`MODEL_ATTRIBUTION.md`. CompassCart source code and generated competition
artifacts remain subject to the team's chosen submission license.
