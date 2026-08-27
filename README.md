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
`QuestionPolicy` asks only when expected conversion gain is positive.
`ResponseBuilder` deduplicates and validates all identifiers.

Additional rationale is in `reports/final/architecture.md`.

## Measured Results

All measurements use the unchanged official evaluator and frozen public data.
The starter baseline scored `0.106710`. The final stable runtime candidate is
commit `c0d444fa`. Its official 200-sample public evaluation scored `0.660411`
(HitRate@10 `0.84`, MRR `0.376036`, MTTC `4.62`, efficiency `0.638`). Scenario
HitRate@10 was `0.90` Boundary, `0.85` Browsing, `0.8625` Buying, and `0.733333`
Intent Override. The result exactly matches the previous stable measurement at
`b641ff97`, so the final hardening delta is `0.000000`.

The one sealed audit scored `0.500563` on 394 representative samples with zero
fallback and zero invalid responses. The final three-trial resource benchmark
also passed with Dense available and zero fallback: P95 was `183.692 ms`,
maximum latency was `529.531 ms`, initialization was `13219.807 ms`, and peak
working set was `557.008 MiB`. P95 improved `58.070%` against the compatible R0
benchmark. S1 through S4 were rejected by their development gates and reverted;
S5 was deferred for accelerated stable delivery. Full aggregate evidence is in
`reports/final/final-results.json` and
`reports/final/score-results-c0d444fa-2026-08-27.json`.

The full automated suite passed 891 tests with 7 skipped. All 9 frozen-input
checks and all 51 delivery-contract checks passed, and Ruff lint passed. macOS
verification is pending.

The agent reports zero prompt and completion tokens. Official runtime API cost
is USD 0.00 per session and USD 0.00 for the full 800-session private set.
Development experiments also used no paid API calls; local asset generation was
a one-time CPU process.

## Limitations

- Intent Override remains the least stable scenario because terse replacement
  messages may expose only one attribute.
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
