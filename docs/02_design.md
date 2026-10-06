# CMN-C1-066 GovernmentProcedureQAAgent — Design Document

**Template ID:** CMN-C1-066
**Stage:** ② Design
**Author:** F.Liu
**Date:** 2026-06-02
**Parent DRAFT:** (internal reference removed)

---

## 1. Overview

A citizen/SMB-facing **FAQ agent for Japanese government administrative procedures**. A natural-language question ("個人事業廃業届の必要書類は？", "建設業許可の処理期間は？") is embedded, matched against the **e-Gov public corpus** and a tenant-configurable **municipal supplement KB**, and answered with three structured items: **必要書類 (required documents) / 処理期間 (processing time) / 提出窓口 (submission channel)**, with source citations.

This is a retrieval-augmented template: it answers from records it looked up, not from a model's own knowledge.

| Aspect | Value |
|---|---|
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Pattern | question → retrieve → compose |
| Nodes | four, composed explicitly onto the fixed backbone |

The agent is built on `AgentBaseGraph` with four explicit nodes rather than by inheriting a single monolithic run method, so each stage of the pipeline is separately testable and separately gated.

---

## 2. Architecture

### 2.1 Node Topology

```
QueryEmbed            ← refusal screens; strip markup; build the term set
    ↓
eGovRetrieve          ← retrieve top-k from the e-Gov public corpus
    ↓
MunicipalKBRetrieve   ← retrieve top-k from the tenant municipal KB (optional)
    ↓
AnswerGenerate        ← LLM answer (必要書類/処理期間/提出窓口) + INDETERMINATE for eligibility
```

### 2.2 Node Descriptions

| Node | Responsibility | Config Keys |
|------|----------------|-------------|
| QueryEmbed | **Refusal screens, before anything is looked up** — a question carrying an individual identification number, or a chat-template control token, is refused here. Strip markup; bound the length; produce the term set retrieval scores against. | `enable_mynumber_block` |
| eGovRetrieve | Score the query against the e-Gov corpus (`egov_corpus_path` JSONL); return top-k procedure records. | `egov_corpus_path`, `retrieval.top_k`, `retrieval.score_threshold` |
| MunicipalKBRetrieve | Same against the tenant municipal KB (`municipal_kb_path`); optional — empty/missing KB is allowed (e-Gov-only answer). | `municipal_kb_path`, `retrieval.top_k` |
| AnswerGenerate | Extracts the three structured fields and citations from the retrieved records. An injected model client writes the natural-language summary only; the structured fields are never model-generated. **Eligibility questions → INDETERMINATE**, never an individual determination. No retrieval → direct the user to a service counter. | `eligibility_mode` |

> The mandatory output gate is applied at the **agent level**, via `_security_gate_output` — not as a graph node, because a gate that is a node can be routed around. Refusal happens at the QueryEmbed entry, before retrieval.

### 2.3 Inheritance Model

```python
class GovernmentProcedureQAAgent(AgentBaseGraph):
    """Question answering over administrative-procedure records.

    Direct inheritance from the framework base class; the pipeline is
    question -> retrieve -> compose.
    """

    def register_nodes(self) -> None:
        self.add_node("QueryEmbed", QueryEmbedNode(...))
        self.add_node("eGovRetrieve", EGovRetrieveNode(...))
        self.add_node("MunicipalKBRetrieve", MunicipalKBRetrieveNode(...))
        self.add_node("AnswerGenerate", AnswerGenerateNode(...))

    def add_edges(self) -> None:
        self.add_edge("QueryEmbed", "eGovRetrieve")
        self.add_edge("eGovRetrieve", "MunicipalKBRetrieve")
        self.add_edge("MunicipalKBRetrieve", "AnswerGenerate")

    def _security_gate_output(self, output: str, config=None) -> str:
        """Mandatory output gate — do not disable."""
        return _apply_s3(str(output))
```

The node entry point is `execute(self, state) -> dict`, returning a partial state dict.

### 2.4 State Schema (`src/schemas/state.py`)

Flat, JSON-string payloads so the state stays serializable across process boundaries. Plain inheritance from the framework `AgentState`.

```python
from typing import Optional
from framework.schemas.agent_state import AgentState

class GovernmentProcedureQAState(AgentState):
    # Input
    query: Optional[str]                 # natural-language question
    municipality: Optional[str]          # tenant municipality selector (optional)
    # Computed
    query_terms: Optional[str]           # JSON: normalized term set / embedding repr (QueryEmbed)
    egov_hits: Optional[str]             # JSON: top-k e-Gov records (eGovRetrieve)
    municipal_hits: Optional[str]        # JSON: top-k municipal records (MunicipalKBRetrieve)
    answer: Optional[str]                # final structured answer JSON (必要書類/処理期間/提出窓口 + citations)
    blocked: Optional[str]               # set when the question is refused: reason string
```

ARCH-01: import `AgentState` from `framework.schemas.agent_state`; no local `AgentState` shadow.

---

## 3. Configuration Schema (`config/agent.yaml`)

```yaml
agent_id: CMN-C1-066
name: GovernmentProcedureQAAgent
version: "0.1.0"
category: C1
industry: CMN
pattern: rag
level2_base: VectorRAGAgent       # conceptual reference only; no import

node_flow:
  - QueryEmbed
  - eGovRetrieve
  - MunicipalKBRetrieve
  - AnswerGenerate

llm:
  model: gpt-4o
  max_tokens: 1024
  temperature: 0.2                # factual Q&A; grounded in retrieved context

retrieval:
  top_k: 5
  score_threshold: 0.2            # below → low-confidence → recommend 窓口

# Corpora (JSONL). e-Gov is public/zero-cost; municipal KB is tenant-configurable.
egov_corpus_path: config/egov_procedures.jsonl
municipal_kb_path: config/municipal_supplement.jsonl

enable_mynumber_block: true       # identifier hard block — do not disable
eligibility_mode: "indeterminate" # eligibility questions always return INDETERMINATE

security:
  classification: MEDIUM
  pii_fields: []                  # public info only; no personal data stored
  s3_gate_enabled: true           # MANDATORY
```

---

## 4. Domain-Specific Data

### 4.1 Corpus format (`egov_procedures.jsonl`, `municipal_supplement.jsonl`)

One JSON object per line:

```json
{"procedure_id": "kosei-haigyo-todoke", "title": "個人事業の廃業届出書", "keywords": ["廃業","個人事業","届出"], "required_documents": ["廃業届", "青色申告取りやめ届出書(任意)"], "processing_time": "即日(提出のみ)", "submission_channel": "所轄税務署 / e-Tax", "source": "e-Gov:法人番号法 ..."}
```

Retrieval is **in-memory term-overlap scoring** over the JSONL (no external vector store / live API for the MVP — deterministic and testable). `retrieval.score_threshold` gates low-confidence matches. The retriever is injected into the retrieve nodes (constructor arg), so tests pass a small fixture corpus.

### 4.2 Identification-number hard block

`QueryEmbed` refuses any question carrying an individual-number pattern (a bare 12-digit run, or the 個人番号 / マイナンバー keywords) while `enable_mynumber_block: true`. It sets `state["blocked"]` with a safe reason and short-circuits: **no retrieval, no model call**. The reply directs the user not to send identification numbers to this service.

### 4.2b Control-token screen

The same node refuses a question carrying a chat-template control token (`<|...|>`, `[INST]`, `<<SYS>>`). The screen runs on the raw question **and** on the markup-stripped form: stripping `<...>` spans would otherwise delete the token and forward the directive it wrapped, turning a recognizable attack into ordinary-looking text.

### 4.2c Caller-supplied municipality

`municipality` arrives from the caller. It is validated at the pipeline boundary — a string, at most 64 characters, over letters, digits, spaces and hyphens only — and rejected otherwise. The rejected value is never echoed back; the error names the field.

### 4.3 Eligibility → INDETERMINATE

When `eligibility_mode: "indeterminate"`, AnswerGenerate must **not** make an individual eligibility determination ("あなたは該当します/しません"). Eligibility-type questions get an `INDETERMINATE` answer that explains the general rule and directs the user to the responsible 窓口. The agent answers *what the procedure is*, never *whether this individual qualifies*.

---

## 5. Security

| Concern | Implementation | Behavior |
|---------|----------------|----------|
| Input trust and refusal | `QueryEmbed`, plus the declared `required_trust_level` | Identification-number hard block; control-token screen on raw and stripped forms; length bound; markup strip |
| Caller-parameter contract | `validate_municipality` at the pipeline boundary | Type, length and alphabet enforced; rejected values never echoed |
| Structured-parameter screen | `src/api/server.py` | `input_context` is screened for credential-shaped values before `invoke()`, using the framework's own detector, and refused with a 400 naming the field |
| Tenant scoping | supplement-corpus filter | Published information only; no personal data stored; a supplement record tagged for another municipality is excluded |
| Output filtering | `GovernmentProcedureQAAgent._security_gate_output()` | Framework credential detection plus two supplementary shapes; INDETERMINATE enforced for eligibility; `s3_gate_enabled` must stay `true` |
| Containment | `GovernmentProcedureQAAgent.get_output()` | A non-successful run surfaces identifiers and status only — every content field is withheld, including the retrieved records the refused answer was built from |
| Audit logging | `emit_trace_event()` | Counts and outcome labels only: never question text, retrieved records or answer text |
| Credentials | none required | The manifest declares no secrets; nothing in `src/` carries a credential |

### 5.0 What the output invariant is, and is not

This agent renders **no monetary aggregates and no numeric estimates**. Every
field it emits — required documents, processing time, submission channel,
citations — is text copied from a retrieved record. There is therefore no
rounding or precision grid to enforce, and no numeric snap that could mangle an
identifier or a decimal on the way out.

The output invariants it *does* enforce are the two below: no credential-shaped
content leaves the agent, and no individual eligibility determination is ever
produced. Both are enforced in code and tested in both directions.

### 5.1 Invariants
- `enable_mynumber_block: true` and `s3_gate_enabled: true`. The agent refuses to construct when the output gate is disabled.
- Eligibility answers are INDETERMINATE — never individual determinations.
- Published corpus only; no identification numbers or personal data persisted in state.
- A run that does not succeed carries no content out.
- No direct platform-SDK import in `src/`.

---

## 6. Test Specification (surface)

Full TC + PB matrix in `docs/03_test_spec.md`.

### 6.1 Test Cases (TC)

| ID | Scope |
|----|-------|
| TC-01 | State contract — framework `AgentState` import, no shadow, JSON-string fields |
| TC-02 | Known procedure question → correct 必要書類/処理期間/提出窓口 + citation |
| TC-03 | No corpus match (low confidence) → 窓口 recommendation, no hallucinated answer |
| TC-04 | **Identification-number question → hard block** (refusal; no retrieval, no model call) |
| TC-05 | **Eligibility question → INDETERMINATE** (no individual determination) |
| TC-06 | Municipal KB supplements e-Gov (tenant municipality) |
| TC-07 | Empty/missing municipal KB → e-Gov-only answer (graceful) |
| TC-08 | CJK query + answer preserved (no mojibake) |

### 6.2 Proof-of-Boundary (PB)

| ID | Boundary | Expected |
|----|----------|----------|
| PB-1 | Refusal precedes retrieval | 12-digit / 個人番号 question refused before retrieve; model not called |
| PB-2 | Output gate | every credential shape withheld; ordinary answers byte-identical |
| PB-2b | Answer-node scan | a credential reaching the rendered answer raises rather than being emitted |
| PB-2c | Containment | a non-successful run carries no retrieved records, no answer, no traceback |
| PB-3 | INDETERMINATE invariant | an eligibility question never yields an individual determination |
| PB-4 | Import isolation | no direct platform-SDK import in `src/` (AST scan) |
| PB-5 | State serialization | post-pipeline state is primitives and JSON strings only |
| PB-6 | Invoke order | QueryEmbed → eGovRetrieve → MunicipalKBRetrieve → AnswerGenerate; gate at agent level |

---

## 7. Dependencies

- `agenticstar-agentcore` — the framework; installed, not vendored
- Used from it: `framework.graph.agent_base_graph.AgentBaseGraph`, `framework.nodes.BaseNode`, `framework.nodes.function_node.FunctionNode`, `framework.schemas.agent_state.AgentState`, `framework.schemas.trust_level.TrustLevel`, `framework.security.credential_detector.detect_credentials`, `framework.utils.config_loader.load_agent_config`, `shared.utils.audit_logger.emit_trace_event`
- `pyyaml` — configuration loading
- pytest, pytest-asyncio, ruff, mypy (dev)
- Retrieval is in-memory over JSONL corpora — **no external vector store**. Swapping in a vector store or a live API is the natural first extension.

---

## 8. Stage Gate Checklist (Design → Implementation)

- [x] `docs/02_design.md` committed
- [x] L1 Base row present
- [x] Design matches the shipped code: `src/schemas/state.py`, `src/nodes/*.py`, `src/graph/graph.py`, `src/api/server.py`, `config/agent.yaml` (manifest), `config/config.yaml` (runtime), `config/egov_procedures.jsonl`, `config/municipal_supplement.jsonl`
