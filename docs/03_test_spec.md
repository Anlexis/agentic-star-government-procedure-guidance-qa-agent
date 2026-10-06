# CMN-C1-066 GovernmentProcedureQAAgent — Test Specification

**Template ID:** CMN-C1-066
**Stage:** ④ Test
**Author:** F.Liu
**Date:** 2026-06-03
**Implements:** (internal reference removed) · design [docs/02_design.md §6](02_design.md)

---

## 1. Scope & Environment

Verifies the four-node RAG pipeline (`QueryEmbed → eGovRetrieve → MunicipalKBRetrieve →
AnswerGenerate`) and its security invariants against the design (§2–§5), plus the
framework-compliance and Proof-of-Boundary requirements.

- **Runner:** `pytest` (`pytest==8.1.1`, `pytest-asyncio` strict).
- **Framework:** provided by the installed `agenticstar-agentcore` package. No
  direct platform-SDK import is exercised.
- **Determinism:** tests inject a small fixture corpus and a spy/None LLM client
  (`tests/conftest.py`) — they do **not** depend on `config/*.jsonl` or any live
  service, so results are deterministic and hermetic.
- **Commands:**
  ```bash
  python -m pytest tests/ -v
  python -m pytest tests/proof_of_boundary/ -v
  ```

> **Retrieval note:** term-overlap retrieval does not segment Japanese text, so
> test queries use space-separated salient terms (e.g. `廃業 個人事業 届出`) that
> match the corpus `keywords`. A full sentence retrieves nothing. Adding a
> segmenter or a vector store is the natural first extension, and would change
> what these queries need to look like.

---

## 2. Test Cases (TC)

| ID | Title | Preconditions | Steps | Expected |
|----|-------|---------------|-------|----------|
| **TC-01** | State contract | — | Import `GovernmentProcedureQAState` | Subclasses the framework `AgentState` without shadowing it; declares `query / municipality / query_terms / egov_hits / municipal_hits / answer / blocked` |
| **TC-02** | Known procedure query | fixture corpus | Run agent with `廃業 個人事業 届出` | `status=answered`; `required_documents` / `processing_time` / `submission_channel` populated from the matched record; ≥1 citation |
| **TC-03** | No corpus match | fixture corpus | Run with an out-of-domain query | `status=low_confidence`; empty `required_documents`; no citations; 窓口 notice; **no hallucinated fields** |
| **TC-04** | Identification-number block | spy model client | Run with a 12-digit identification-number query | `blocked` set; **no `answer`**; retrieval empty (`egov_hits=="[]"`); **model never called** |
| **TC-05** | Eligibility → INDETERMINATE | fixture corpus | Run with "私は…対象になりますか" | `status=indeterminate`; `eligibility=INDETERMINATE`; no individual 該当/非該当 determination |
| **TC-06** | Municipal override | fixture corpus (e-Gov + municipal sharing `procedure_id`) | Run with `転入 引越 住民登録` + `municipality=渋谷区` | Municipal record supplements/overrides e-Gov (`submission_channel` = 渋谷区; municipal-only doc present); both origins cited |
| **TC-07** | Empty / missing municipal KB | e-Gov fixture only | Run `転入 引越 住民登録`, no municipal corpus | `status=answered` from e-Gov only; graceful (no crash); only e-Gov citation |
| **TC-08** | CJK preserved | fixture corpus | Run a CJK query | Answer JSON round-trips with CJK intact (no mojibake / `\u` escaping) |

---

## 3. Proof-of-Boundary (PB)

| ID | Boundary | Steps | Expected |
|----|----------|-------|----------|
| **PB-1** | Refusal precedes retrieval and generation | Run 12-digit + `個人番号` queries with a spy model client | Blocked before retrieval; `egov_hits`/`municipal_hits` empty; model call count == 0 |
| **PB-2** | Output gate | Call `agent._security_gate_output()` with each credential shape | The credential never appears in the returned text; ordinary answers pass through byte-identical |
| **PB-2b** | Answer-node scan | Drive a poisoned record through `AnswerGenerateNode.execute()` | Raises rather than emitting; a clean record answers normally |
| **PB-2c** | Containment on the error path | Poison a record so the gate refuses, then read what the caller receives | No retrieved records, no answer, no traceback, no source paths — identifiers and status only |
| **PB-3** | INDETERMINATE invariant | Eligibility query end-to-end | Output never contains an individual 該当/非該当 determination; `eligibility=INDETERMINATE` |
| **PB-4** | Import isolation | AST-scan every `src/**/*.py` | No direct platform-SDK import |
| **PB-5** | State serialization | Run pipeline; inspect final state | All values are primitives / JSON strings; `json.dumps(state)` succeeds |
| **PB-6** | Invoke order and gate placement | Instrument node order; inspect class | Execution `QueryEmbed → eGovRetrieve → MunicipalKBRetrieve → AnswerGenerate`; `_security_gate_output` is on the **agent**, never a node |

---

## 4. Traceability

| Spec | Test file · function |
|------|----------------------|
| TC-01…TC-08 | `tests/test_government_procedure_qa.py::test_tc01_…tc08_…` |
| PB-1…PB-6 | `tests/proof_of_boundary/test_pb_boundary.py::test_pb1_…pb6_…` |
| Caller contract, refusal screens, containment | `tests/unit/test_caller_contract.py` |
| Structured-parameter credential screen | `tests/unit/test_input_context_screen.py` |
| Import isolation, invoke order, state safety | `tests/proof_of_boundary/test_import_isolation.py`, `test_pb_invoke_order.py`, `test_state_safety.py` |
| Gate-override compliance | `tests/unit/test_framework_compliance_tc06_tc07.py` |
| Shared fixtures (corpus, agent factory, spy model client) | `tests/conftest.py` |

The CI pipeline additionally enforces import isolation, credential scanning,
trust-level declaration, audit-trace coverage, manifest schema, dependency
pinning and license compliance.

---

## 5. Results

| Suite | Status |
|-------|--------|
| `tests/` (TC) | ✅ all pass (run locally; CI `run-tests`) |
| `tests/proof_of_boundary/` (PB) | ✅ all pass |

> Once this spec is on `develop`, the CI `gate-test` activates legitimately and
> must stay green (the prior pre-Stage-④ red on `gate-test` — (internal reference removed) — is resolved by this MR).
