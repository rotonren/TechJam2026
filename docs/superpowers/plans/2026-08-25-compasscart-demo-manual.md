# CompassCart Demo Chat and User Manual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a novice-friendly terminal demo for CompassCart and a Chinese Word manual that teaches live use and judging presentation.

**Architecture:** `tools/demo_chat.py` is a presentation adapter around the unchanged `agent.Agent`; it reads catalog metadata already held by the Agent and prints compact results and trace evidence. A separate DOCX builder uses the existing document helpers and does not affect runtime behavior.

**Tech Stack:** Python 3.10+, argparse, standard-library terminal I/O, existing CompassCart Agent, pytest, python-docx.

---

### Task 1: Specify the terminal formatter and scenarios

**Files:**
- Create: `tests/unit/test_demo_chat.py`
- Create: `tools/demo_chat.py`

- [ ] **Step 1: Write failing formatter and scenario tests**

Test that prices, missing values, long titles, recommendation numbering, four
scenario names, and trace fields produce readable stable text.

- [ ] **Step 2: Run the focused test and confirm import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_demo_chat.py -v`

Expected: FAIL because `tools.demo_chat` does not exist.

- [ ] **Step 3: Implement minimal pure formatting functions**

Implement `SCENARIOS`, `format_product`, `format_response`, and
`format_trace`. Keep terminal text ASCII-safe except product catalog text.

- [ ] **Step 4: Run focused tests**

Expected: all formatter and scenario tests pass.

### Task 2: Implement interactive and guided execution

**Files:**
- Modify: `tests/unit/test_demo_chat.py`
- Modify: `tools/demo_chat.py`

- [ ] **Step 1: Write failing tests for commands and turn handling**

Use a fake Agent to test `/new`, `/trace`, `/quit`, EOF, the ten-turn guard,
and scripted scenario execution without loading the real catalog.

- [ ] **Step 2: Run focused tests and confirm failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_demo_chat.py -v`

- [ ] **Step 3: Implement CLI parsing and session loop**

Add `--catalog`, `--scenario`, `--lexical`, `--no-trace`, and `--top`.
Construct the Agent once, handle missing catalog with exit code 2, and keep all
recommendation behavior delegated to the existing Agent.

- [ ] **Step 4: Run focused and full quality gates**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_demo_chat.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check tools/demo_chat.py tests/unit/test_demo_chat.py
.\.venv\Scripts\python.exe -m ruff format --check tools/demo_chat.py tests/unit/test_demo_chat.py
```

Expected: all commands pass.

### Task 3: Verify real guided demonstrations

**Files:**
- No source changes expected.

- [ ] **Step 1: Run all four scenarios with lexical fallback**

Run each `--scenario` with `--lexical` against `data/catalog.jsonl` and verify
non-empty legal recommendations, zero tokens, and clean exit.

- [ ] **Step 2: Run override with dense enabled**

Verify intent version increments and the final recommendation set changes from
the dress intent to the belt intent.

### Task 4: Create the product usage and showcase DOCX

**Files:**
- Create: `dist/docx_work/build_product_user_manual.py`
- Create: `dist/CompassCart_产品使用与展示手册_CN.docx`

- [ ] **Step 1: Build a compact-reference-guide DOCX**

Include preflight, one-command launch, screen legend, free chat, four guided
scenarios, three-minute script, fallback demonstration, troubleshooting, and a
rehearsal checklist. Exclude development and team workflow content.

- [ ] **Step 2: Perform structural and WPS compatibility checks**

Verify OOXML opens, headings and required phrases exist, tables use fixed DXA
geometry, and WPS can open the file. Do not create PNGs because the user
explicitly declined PNG rendering.

- [ ] **Step 3: Run final repository gates**

Run full pytest and scoped Ruff checks, then deliver only the new Word manual.
