"""CMN-C1-066 — Test Cases TC-01…TC-08 (see docs/03_test_spec.md §2).

Re-targeted onto the released ``agenticstar-agentcore`` SDK: the agent is
exercised end-to-end via ``.invoke()`` (the ``run_agent`` conftest helper) —
``.run()`` does not exist in the released SDK. The query is passed as
``user_input`` and the municipality via ``input_context``; the output dict is
produced by ``GovernmentProcedureQAAgent.get_output``.
"""

import json

from framework.schemas.agent_state import AgentState

from src.schemas.state import GovernmentProcedureQAState
from tests.conftest import run_agent


# ── TC-01: State contract ────────────────────────────────────


def test_tc01_state_contract():
    # Extends the framework AgentState (ARCH-01: no local shadow). The released
    # AgentState is a real TypedDict, so issubclass() raises and __orig_bases__ is
    # CPython-version-dependent (populated on 3.13, empty on 3.11). Assert the
    # relationship via the TypedDict key sets (stable API since 3.9): a subclass's
    # __required_keys__/__optional_keys__ include the parent's keys.
    parent_keys = set(AgentState.__required_keys__) | set(AgentState.__optional_keys__)
    child_keys = set(GovernmentProcedureQAState.__required_keys__) | set(GovernmentProcedureQAState.__optional_keys__)
    assert parent_keys and parent_keys <= child_keys, "State must extend framework AgentState"
    assert GovernmentProcedureQAState.__module__ == "src.schemas.state"
    for field in (
        "query",
        "municipality",
        "query_terms",
        "egov_hits",
        "municipal_hits",
        "answer",
        "blocked",
    ):
        assert field in child_keys, f"missing state field: {field}"


# ── TC-02: Known procedure query ─────────────────────────────


def test_tc02_known_procedure_query(make_agent):
    agent = make_agent()
    answer = json.loads(run_agent(agent, "廃業 個人事業 届出")["answer"])
    assert answer["status"] == "answered"
    assert answer["required_documents"] == ["個人事業の開業・廃業等届出書"]
    assert answer["processing_time"] and answer["submission_channel"]
    assert len(answer["citations"]) >= 1
    assert answer["citations"][0]["procedure_id"] == "kojin-haigyo"


# ── TC-03: No corpus match → low confidence, no hallucination ─


def test_tc03_no_match_low_confidence(make_agent):
    agent = make_agent()
    answer = json.loads(run_agent(agent, "宇宙旅行 予約 方法")["answer"])
    assert answer["status"] == "low_confidence"
    assert answer["required_documents"] == []
    assert answer["citations"] == []
    assert answer.get("notice")


# ── TC-04: identification-number hard block ──────────────────


def test_tc04_mynumber_hard_block(make_agent, spy_llm):
    agent = make_agent(llm=spy_llm)
    out = run_agent(agent, "私のマイナンバーは123456789012です 廃業 届出")
    assert out.get("blocked")  # refusal reason set
    assert "answer" not in out  # no answer generated
    assert out.get("egov_hits") == "[]"  # retrieval produced nothing
    assert spy_llm.calls == []  # LLM never called


# ── TC-05: Eligibility → INDETERMINATE ───────────────────────


def test_tc05_eligibility_indeterminate(make_agent):
    agent = make_agent()
    answer = json.loads(run_agent(agent, "私は 建設業 許可 の対象になりますか")["answer"])
    assert answer["status"] == "indeterminate"
    assert answer["eligibility"] == "INDETERMINATE"


# ── TC-06: Municipal override ────────────────────────────────


def test_tc06_municipal_override(make_agent):
    agent = make_agent()
    answer = json.loads(run_agent(agent, "転入 引越 住民登録", municipality="渋谷区")["answer"])
    assert answer["status"] == "answered"
    assert "渋谷区" in answer["submission_channel"]  # municipal override
    assert any("渋谷" in d for d in answer["required_documents"])  # municipal supplement
    origins = {c["origin"] for c in answer["citations"]}
    assert "municipal" in origins and "egov" in origins


# ── TC-07: Empty / missing municipal KB → e-Gov-only ─────────


def test_tc07_no_municipal_kb_graceful(make_agent):
    agent = make_agent(municipal=[])
    answer = json.loads(run_agent(agent, "転入 引越 住民登録", municipality="渋谷区")["answer"])
    assert answer["status"] == "answered"
    assert answer["submission_channel"] == "市区町村役場"  # e-Gov record
    assert all(c["origin"] == "egov" for c in answer["citations"])


# ── TC-08: CJK preserved (no mojibake) ───────────────────────


def test_tc08_cjk_preserved(make_agent):
    agent = make_agent()
    raw = run_agent(agent, "廃業 個人事業 届出")["answer"]
    # Stored as ensure_ascii=False JSON — CJK must be literal, not \u-escaped.
    assert "個人事業の開業・廃業等届出書" in raw
    assert "\\u" not in raw
    answer = json.loads(raw)
    assert "廃業" in answer["citations"][0]["title"]
