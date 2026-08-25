# CompassCart Demo Chat and User Manual Design

## Purpose

Give a non-developer one reliable command for using CompassCart as a live,
multi-turn shopping assistant and a short Chinese Word manual for presenting
the product to judges. This work is demonstration-only and must not change the
scored Agent, ranking configuration, evaluator, catalog, or frozen submission
package.

## Chosen Approach

Add `tools/demo_chat.py` as a thin terminal adapter around the existing
`agent.Agent`. The adapter owns terminal input/output only. It creates one
Agent, starts or resets a session, calls `respond(..., top_k=10)`, and formats
the first five recommendations with catalog title, price, rating, and ASIN.
The existing Agent remains the single source of recommendation behavior.

The alternative of documenting inline Python was rejected because it is too
fragile for novices and cannot support free conversation cleanly. A web UI was
rejected because it adds dependencies and deployment risk without improving
the competition's required Agent contract.

## User Experience

The primary command is:

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m tools.demo_chat --catalog data/catalog.jsonl
```

Startup states that loading 50,000 products can take about 15 seconds. The
terminal then accepts English shopping requests. Each response shows:

- the Agent's natural-language question;
- the first five recommendations with readable product metadata;
- optional demo evidence: route, intent version, active constraints,
  candidate count, latency, fallback list, dense availability, and zero token
  usage.

Commands are `/help`, `/new`, `/trace`, and `/quit`. A session stops accepting
turns after turn 10 until `/new` is used. EOF and Ctrl+C exit cleanly.

## Guided Scenarios

`--scenario` runs one of four rehearsable scripts and exits:

- `browsing`: broad request, then a preference, showing route refinement;
- `buying`: explicit product, color, use case, and budget;
- `override`: dress intent replaced by a black leather belt intent;
- `boundary`: no-preference replies without repeating the rejected attribute.

The manual warns that exact product IDs may vary between dense and lexical
mode. The observable contract is the scenario behavior, legal unique catalog
IDs, state change, and zero tokens.

## Error Handling

Missing catalog returns a short actionable error and exit code 2. Agent load or
turn errors are reported without a traceback by default. `--lexical` sets
`COMPASSCART_DISABLE_DENSE=1` before Agent construction. Interactive input in
Chinese is discouraged because the current deterministic parser is optimized
for English competition language.

## Verification

Unit tests cover argument parsing, product formatting, trace formatting,
scenario definitions, session reset, turn limit, EOF, and clean error output.
The full repository test suite and scoped Ruff checks must remain green. The
four guided scenarios are smoke-tested against the frozen 50,000-product
catalog in lexical mode, and at least the override scenario is checked with
dense enabled.

## Word Deliverable

Create `dist/CompassCart_产品使用与展示手册_CN.docx` as a compact reference
guide. It contains only product startup, screen interpretation, live usage,
four guided demos, a three-minute run-of-show, offline fallback, troubleshooting,
and a final rehearsal checklist. It explicitly excludes team assignment,
development CV, Git, and packaging procedures.
